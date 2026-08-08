/**
 * train_tinydrone.c — tinycml CNN training for military target detection
 *
 * Build: make train_tinydrone
 * Run:   ./build/bin/train_tinydrone [dataset_dir] [epochs]
 *
 * Uses tinycml's own CNN + backward pass — zero Python dependencies.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <dirent.h>
#include <time.h>
#include "cnn.h"
#include "matrix.h"

/* Stub for PNG loading — use stb_image or similar in production */
/* For now: load pre-processed 32x32 raw RGB data */

#define IMG_H 32
#define IMG_W 32
#define IMG_C 3
#define N_CLASSES 4
#define BATCH_SIZE 32
#define LEARNING_RATE 0.001

static const char *CLASS_NAMES[] = {"tank", "armored_vehicle", "drone_uav", "background"};

/* ============================================
 * PNG Loading via external library
 * ============================================ */

/* Simple PNG reader using stb_image — include at compile time */
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

/**
 * Load a PNG image and return as a 1-row Matrix (1 × C*H*W).
 * Normalizes pixel values to [-1, 1].
 */
static Matrix* load_image(const char *path) {
    int w, h, channels;
    unsigned char *data = stbi_load(path, &w, &h, &channels, 3);  /* Force RGB */
    if (!data) return NULL;

    Matrix *m = matrix_alloc(1, (size_t)IMG_C * IMG_H * IMG_W);
    if (!m) { stbi_image_free(data); return NULL; }

    for (int y = 0; y < IMG_H; y++) {
        for (int x = 0; x < IMG_W; x++) {
            for (int c = 0; c < IMG_C; c++) {
                int src_idx = (y * w + x) * 3 + c;
                int dst_idx = c * IMG_H * IMG_W + y * IMG_W + x;
                m->data[dst_idx] = (double)data[src_idx] / 127.5 - 1.0;  /* [-1, 1] */
            }
        }
    }

    stbi_image_free(data);
    return m;
}

/* ============================================
 * Dataset Loading
 * ============================================ */

typedef struct {
    Matrix **images;
    int *labels;
    int count;
    int capacity;
} Dataset;

static Dataset* dataset_load(const char *dir_path, int class_idx) {
    Dataset *ds = calloc(1, sizeof(Dataset));
    ds->capacity = 1000;
    ds->images = malloc((size_t)ds->capacity * sizeof(Matrix*));
    ds->labels = malloc((size_t)ds->capacity * sizeof(int));

    DIR *dir = opendir(dir_path);
    if (!dir) { free(ds); return NULL; }

    struct dirent *entry;
    while ((entry = readdir(dir))) {
        if (!strstr(entry->d_name, ".png")) continue;

        char path[1024];
        snprintf(path, sizeof(path), "%s/%s", dir_path, entry->d_name);

        Matrix *img = load_image(path);
        if (!img) continue;

        if (ds->count >= ds->capacity) {
            ds->capacity *= 2;
            ds->images = realloc(ds->images, (size_t)ds->capacity * sizeof(Matrix*));
            ds->labels = realloc(ds->labels, (size_t)ds->capacity * sizeof(int));
        }
        ds->images[ds->count] = img;
        ds->labels[ds->count] = class_idx;
        ds->count++;
    }
    closedir(dir);
    return ds;
}

static void dataset_free(Dataset *ds) {
    if (!ds) return;
    for (int i = 0; i < ds->count; i++) matrix_free(ds->images[i]);
    free(ds->images);
    free(ds->labels);
    free(ds);
}

/* ============================================
 * Loss function: Cross-Entropy
 * ============================================ */

static double cross_entropy_loss(const Matrix *probs, const int *labels, int n) {
    double loss = 0.0;
    for (int i = 0; i < n; i++) {
        int cls = labels[i];
        double p = probs->data[(size_t)i * N_CLASSES + cls];
        loss -= log(p + 1e-8);
    }
    return loss / n;
}

static Matrix* cross_entropy_grad(const Matrix *probs, const int *labels, int n) {
    Matrix *grad = matrix_copy(probs);
    if (!grad) return NULL;
    for (int i = 0; i < n; i++) {
        grad->data[(size_t)i * N_CLASSES + labels[i]] -= 1.0;
    }
    return grad;
}

/* ============================================
 * CNN Forward + Backward (manual, without Estimator)
 * ============================================ */

/**
 * Manually run forward+backward through CNN blocks.
 * This is needed because Estimator doesn't support training yet.
 */
static Matrix* cnn_forward_backward(CNNModel *net, const Matrix *input,
                                     Matrix **cache_outputs) {
    /* Forward through conv/pool blocks, caching intermediate outputs */
    int n = (int)input->rows;
    int current_h = net->orig_input_h;
    int current_w = net->orig_input_w;
    int current_c = net->orig_input_c;

    Matrix *current = matrix_copy(input);
    if (!current) return NULL;

    for (int i = 0; i < net->n_blocks; i++) {
        CNNBlock *block = net->blocks[i];
        cache_outputs[i] = current;  /* Cache input to this block */
        Matrix *next = NULL;

        switch (block->type) {
            case CNN_LAYER_CONV2D: {
                Conv2D *conv = (Conv2D*)block->layer;
                next = conv2d_forward(conv, current, n, current_h, current_w);
                current_h = conv2d_out_size(current_h, conv->kernel_h, conv->stride_h, conv->pad_h);
                current_w = conv2d_out_size(current_w, conv->kernel_w, conv->stride_w, conv->pad_w);
                current_c = conv->out_channels;
                break;
            }
            case CNN_LAYER_RELU: {
                /* In-place ReLU */
                for (size_t j = 0; j < current->rows * current->cols; j++)
                    if (current->data[j] < 0.0) current->data[j] = 0.0;
                next = current;
                current = NULL;
                break;
            }
            case CNN_LAYER_MAXPOOL: {
                MaxPool2D *pool = (MaxPool2D*)block->layer;
                next = maxpool2d_forward(pool, current, n, current_c, current_h, current_w);
                current_h = pool2d_out_size(current_h, pool->pool_h, pool->stride_h);
                current_w = pool2d_out_size(current_w, pool->pool_w, pool->stride_w);
                break;
            }
            case CNN_LAYER_FLATTEN:
                next = current;
                current = NULL;
                break;
            case CNN_LAYER_DENSE: {
                void **ptr = (void**)block->layer;
                Matrix *w = (Matrix*)ptr[0];
                Matrix *b = (Matrix*)ptr[1];
                next = matrix_matmul(current, w);
                for (int r = 0; r < n; r++)
                    for (int oc = 0; oc < block->output_size; oc++)
                        next->data[(size_t)r * block->output_size + oc] += b->data[oc];
                if (block->activation == ACTIVATION_RELU) {
                    for (size_t j = 0; j < next->rows * next->cols; j++)
                        if (next->data[j] < 0.0) next->data[j] = 0.0;
                }
                break;
            }
            default:
                break;
        }
        if (current && current != next) matrix_free(current);
        current = next;
    }

    return current;  /* Final output */
}

static void cnn_backward_pass(CNNModel *net, Matrix *dout,
                               Matrix **cache_outputs) {
    Matrix *dcurrent = dout;
    Matrix *dprev = NULL;

    for (int i = net->n_blocks - 1; i >= 0; i--) {
        CNNBlock *block = net->blocks[i];

        switch (block->type) {
            case CNN_LAYER_CONV2D: {
                Conv2D *conv = (Conv2D*)block->layer;
                dprev = conv2d_backward(conv, dcurrent);
                conv2d_update_weights(conv, LEARNING_RATE);
                if (dcurrent != dout) matrix_free(dcurrent);
                dcurrent = dprev;
                break;
            }
            case CNN_LAYER_RELU: {
                /* ReLU backward: gradient passes through where input was > 0 */
                Matrix *input = cache_outputs[i];
                for (size_t j = 0; j < dcurrent->rows * dcurrent->cols; j++) {
                    if (input->data[j] <= 0.0) dcurrent->data[j] = 0.0;
                }
                dprev = dcurrent;
                break;
            }
            case CNN_LAYER_MAXPOOL: {
                MaxPool2D *pool = (MaxPool2D*)block->layer;
                dprev = maxpool2d_backward(pool, dcurrent);
                if (dcurrent != dout) matrix_free(dcurrent);
                dcurrent = dprev;
                break;
            }
            case CNN_LAYER_DENSE: {
                void **ptr = (void**)block->layer;
                Matrix *w = (Matrix*)ptr[0];
                Matrix *b = (Matrix*)ptr[1];
                
                /* ReLU backward for dense layer */
                Matrix *input_cache = cache_outputs[i];
                if (block->activation == ACTIVATION_RELU) {
                    for (size_t j = 0; j < dcurrent->rows * dcurrent->cols; j++) {
                        if (input_cache->data[j] <= 0.0) dcurrent->data[j] = 0.0;
                    }
                }
                
                /* d_input = dcurrent × w^T */
                Matrix *wT = matrix_transpose(w);
                dprev = matrix_matmul(dcurrent, wT);
                matrix_free(wT);
                
                /* d_weight = input^T × dcurrent */
                Matrix *inputT = matrix_transpose(input_cache);
                Matrix *dw = matrix_matmul(inputT, dcurrent);
                matrix_free(inputT);
                
                /* d_bias = sum of dcurrent per column */
                for (size_t c = 0; c < b->cols; c++) {
                    double sum = 0.0;
                    for (size_t r = 0; r < dcurrent->rows; r++)
                        sum += dcurrent->data[r * dcurrent->cols + c];
                    b->data[c] -= LEARNING_RATE * sum;
                }
                
                /* Update weights */
                for (size_t j = 0; j < w->rows * w->cols; j++)
                    w->data[j] -= LEARNING_RATE * dw->data[j];
                matrix_free(dw);
                
                if (dcurrent != dout) matrix_free(dcurrent);
                dcurrent = dprev;
                break;
            }
            default:
                break;
        }

        /* Free cached input if it's no longer needed */
        if (cache_outputs[i] && i > 0) {
            /* Keep cache for gradient computation */
        }
    }

    if (dcurrent && dcurrent != dout) matrix_free(dcurrent);
}

/* ============================================
 * Main Training Loop
 * ============================================ */

int main(int argc, char **argv) {
    const char *data_root = argc > 1 ? argv[1] : "training/dataset/processed";
    int epochs = argc > 2 ? atoi(argv[2]) : 20;

    printf("tinydrone — tinycml CNN Training\n");
    printf("================================\n");
    printf("Data: %s\n", data_root);
    printf("Epochs: %d, LR: %.4f, Batch: %d\n\n", epochs, LEARNING_RATE, BATCH_SIZE);

    /* Load training dataset */
    printf("Loading dataset...\n");
    Dataset *datasets[N_CLASSES] = {NULL};
    int total = 0;

    for (int c = 0; c < N_CLASSES; c++) {
        char path[1024];
        snprintf(path, sizeof(path), "%s/train/%s", data_root, CLASS_NAMES[c]);
        datasets[c] = dataset_load(path, c);
        if (datasets[c]) {
            printf("  %s: %d samples\n", CLASS_NAMES[c], datasets[c]->count);
            total += datasets[c]->count;
        }
    }
    printf("  Total: %d\n\n", total);

    if (total == 0) {
        printf("No data found! Exiting.\n");
        return 1;
    }

    /* Build CNN model */
    printf("Building TinyCNN...\n");
    CNNModel *net = cnn_create(IMG_H, IMG_W, IMG_C, N_CLASSES);
    cnn_add_conv2d(net, 16, 3, 3, 1, 1);  /* 32→32, 16ch */
    cnn_add_relu(net);
    cnn_add_maxpool(net, 2, 2, 2);         /* 32→16 */
    cnn_add_conv2d(net, 32, 3, 3, 1, 1);  /* 16→16, 32ch */
    cnn_add_relu(net);
    cnn_add_maxpool(net, 2, 2, 2);         /* 16→8 */
    cnn_add_conv2d(net, 64, 3, 3, 1, 1);  /* 8→8, 64ch */
    cnn_add_relu(net);
    cnn_add_maxpool(net, 2, 2, 2);         /* 8→4 */
    cnn_add_flatten(net);
    cnn_add_dense(net, 128, ACTIVATION_RELU);
    cnn_add_dense(net, N_CLASSES, ACTIVATION_SIGMOID);

    int n_blocks = net->n_blocks;
    printf("  Architecture: %d blocks\n", n_blocks);

    /* Allocate cache for intermediate outputs */
    Matrix **cache = calloc((size_t)n_blocks, sizeof(Matrix*));

    /* Training loop */
    printf("\nTraining %d epochs...\n", epochs);
    srand((unsigned)time(NULL));

    for (int epoch = 0; epoch < epochs; epoch++) {
        double total_loss = 0.0;
        int batches = 0;

        /* Shuffle sample indices */
        int *indices = malloc((size_t)total * sizeof(int));
        int idx = 0;
        for (int c = 0; c < N_CLASSES; c++) {
            if (!datasets[c]) continue;
            for (int i = 0; i < datasets[c]->count; i++) {
                indices[idx++] = c * 100000 + i;  /* Encode class+sample */
            }
        }
        /* Fisher-Yates shuffle */
        for (int i = total - 1; i > 0; i--) {
            int j = rand() % (i + 1);
            int tmp = indices[i];
            indices[i] = indices[j];
            indices[j] = tmp;
        }

        /* Mini-batch SGD */
        for (int start = 0; start < total; start += BATCH_SIZE) {
            int batch_n = (start + BATCH_SIZE <= total) ? BATCH_SIZE : (total - start);
            if (batch_n < 2) continue;

            /* Build batch matrix */
            Matrix *batch_X = matrix_alloc((size_t)batch_n, (size_t)IMG_C * IMG_H * IMG_W);
            int *batch_y = malloc((size_t)batch_n * sizeof(int));

            for (int b = 0; b < batch_n; b++) {
                int enc = indices[start + b];
                int cls = enc / 100000;
                int sid = enc % 100000;

                /* Copy image data */
                Matrix *img = datasets[cls]->images[sid];
                memcpy(&batch_X->data[(size_t)b * IMG_C * IMG_H * IMG_W],
                       img->data, (size_t)IMG_C * IMG_H * IMG_W * sizeof(double));
                batch_y[b] = cls;
            }

            /* Forward pass */
            Matrix *output = cnn_forward_backward(net, batch_X, cache);

            /* Compute loss */
            double loss = cross_entropy_loss(output, batch_y, batch_n);
            total_loss += loss;

            /* Backward pass */
            Matrix *dout = cross_entropy_grad(output, batch_y, batch_n);
            cnn_backward_pass(net, dout, cache);

            /* Cleanup batch */
            matrix_free(batch_X);
            matrix_free(output);
            matrix_free(dout);
            free(batch_y);

            /* Clean up caches */
            for (int i = 0; i < n_blocks; i++) {
                if (cache[i]) { matrix_free(cache[i]); cache[i] = NULL; }
            }

            batches++;
        }

        free(indices);

        if (epoch % 5 == 0 || epoch == epochs - 1) {
            printf("Epoch %3d/%d | Loss: %.4f | Batches: %d\n",
                   epoch + 1, epochs, total_loss / batches, batches);
        }
    }

    /* Save model (export weights as C header) */
    printf("\nTraining complete. Exporting weights...\n");
    /* TODO: export conv/dense weights as C array */

    /* Cleanup */
    free(cache);
    for (int c = 0; c < N_CLASSES; c++) dataset_free(datasets[c]);
    cnn_free(net);

    printf("Done.\n");
    return 0;
}
