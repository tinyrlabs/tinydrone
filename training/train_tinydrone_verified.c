#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <dirent.h>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include "conv2d.h"
#include "pool2d.h"
#include "matrix.h"

static void relu_inplace(Matrix *m) {
    for (size_t i = 0; i < m->rows*m->cols; i++)
        if (m->data[i] < 0) m->data[i] = 0;
}
typedef struct { Matrix *img; int label; } Sample;
#define P(s) do{printf("%s\n",s);fflush(stdout);}while(0)

int main() {
    P("start");
    /* Load 40 samples (10 per class) */
    const char *cls[]={"tank","armored_vehicle","drone_uav","background"};
    Sample *s=NULL; int n=0;
    for(int c=0;c<4;c++){char p[1024];snprintf(p,sizeof(p),"dataset/processed/train/%s",cls[c]);DIR*d=opendir(p);if(!d)continue;struct dirent*e;int cnt=0;while((e=readdir(d))&&cnt<10){if(!strstr(e->d_name,".png"))continue;char fp[2048];snprintf(fp,sizeof(fp),"%s/%s",p,e->d_name);int ww,hh,ch;unsigned char*px=stbi_load(fp,&ww,&hh,&ch,3);if(!px)continue;Matrix*img=matrix_alloc(1,3072);for(int y=0;y<32;y++)for(int x=0;x<32;x++)for(int cc=0;cc<3;cc++)img->data[cc*1024+y*32+x]=px[(y*ww+x)*3+cc]/127.5-1.0;stbi_image_free(px);s=realloc(s,(n+1)*sizeof(Sample));s[n].img=img;s[n].label=c;n++;cnt++;}closedir(d);}
    P("loaded");
    
    /* Layers with CORRECT dimensions:
     * c1: 3→16, 3x3, pad=1 → 32x32
     * p1: 16ch, 32→16
     * c2: 16→32, 3x3, pad=1 → 16x16
     * p2: 32ch, 16→8
     * flat: 32*8*8 = 2048
     */
    Conv2D*c1=conv2d_create(3,16,3,3,1,1,1,1); P("c1");
    MaxPool2D*p1=maxpool2d_create(2,2,2,2); P("p1");
    Conv2D*c2=conv2d_create(16,32,3,3,1,1,1,1); P("c2");
    MaxPool2D*p2=maxpool2d_create(2,2,2,2); P("p2");
    
    int flat=32*8*8; /* 32ch × 8×8 = 2048 */
    Matrix*fcw=matrix_alloc(flat,64),*fcb=matrix_alloc(1,64);
    Matrix*fcw2=matrix_alloc(64,4),*fcb2=matrix_alloc(1,4);
    for(size_t i=0;i<flat*64;i++)fcw->data[i]=((double)rand()/RAND_MAX-0.5)*0.1;
    for(size_t i=0;i<64*4;i++)fcw2->data[i]=((double)rand()/RAND_MAX-0.5)*0.1;
    P("dense");
    
    double lr=0.001;
    for(int ep=0;ep<3;ep++){
        for(int i=n-1;i>0;i--){int j=rand()%(i+1);Sample t=s[i];s[i]=s[j];s[j]=t;}
        double loss_sum=0; int batches=0;
        
        for(int start=0;start<n;start+=4){
            int bn=(start+4<=n)?4:(n-start); if(bn<2)continue;
            Matrix*X=matrix_alloc(bn,3072); int*lbls=malloc(bn*sizeof(int));
            for(int b=0;b<bn;b++){memcpy(&X->data[b*3072],s[start+b].img->data,3072*sizeof(double));lbls[b]=s[start+b].label;}
            
            /* Forward — CORRECT dimensions */
            Matrix*a1=conv2d_forward(c1,X,bn,32,32);        /* (bn, 16*32*32) */
            Matrix*a1r=matrix_copy(a1);relu_inplace(a1r);
            Matrix*a1p=maxpool2d_forward(p1,a1r,bn,16,32,32);/* (bn, 16*16*16) */
            Matrix*a2=conv2d_forward(c2,a1p,bn,16,16);       /* (bn, 32*16*16) */
            Matrix*a2r=matrix_copy(a2);relu_inplace(a2r);
            Matrix*a2p=maxpool2d_forward(p2,a2r,bn,32,16,16);/* (bn, 32*8*8) */
            Matrix*fc1=matrix_matmul(a2p,fcw);               /* (bn, 64) */
            for(int r=0;r<bn;r++)for(int c=0;c<64;c++)fc1->data[r*64+c]+=fcb->data[c];
            relu_inplace(fc1);
            Matrix*out=matrix_matmul(fc1,fcw2);              /* (bn, 4) */
            for(int r=0;r<bn;r++)for(int c=0;c<4;c++)out->data[r*4+c]+=fcb2->data[c];
            for(int r=0;r<bn;r++){double mx=-1e9;for(int c=0;c<4;c++)if(out->data[r*4+c]>mx)mx=out->data[r*4+c];double s=0;for(int c=0;c<4;c++){out->data[r*4+c]=exp(out->data[r*4+c]-mx);s+=out->data[r*4+c];}for(int c=0;c<4;c++)out->data[r*4+c]/=s;}
            
            double loss=0; for(int i=0;i<bn;i++)loss-=log(out->data[i*4+lbls[i]]+1e-8);
            loss_sum+=loss/bn;
            
            /* Gradient */
            Matrix*dout=matrix_copy(out); for(int i=0;i<bn;i++)dout->data[i*4+lbls[i]]-=1.0;
            
            /* Backward FC2 */
            Matrix*fc1T=matrix_transpose(fc1);Matrix*dw2=matrix_matmul(fc1T,dout);matrix_free(fc1T);
            Matrix*w2T=matrix_transpose(fcw2);Matrix*dfc1=matrix_matmul(dout,w2T);matrix_free(w2T);
            for(size_t j=0;j<fcw2->rows*fcw2->cols;j++)fcw2->data[j]-=lr*dw2->data[j];
            for(size_t c=0;c<fcb2->cols;c++){double s=0;for(size_t r=0;r<dout->rows;r++)s+=dout->data[r*fcb2->cols+c];fcb2->data[c]-=lr*s;}
            matrix_free(dw2);
            /* ReLU FC1 backward */
            for(size_t j=0;j<fc1->rows*fc1->cols;j++)if(fc1->data[j]<=0)dfc1->data[j]=0;
            /* Backward FC1 */
            Matrix*a2pT=matrix_transpose(a2p);Matrix*dw1=matrix_matmul(a2pT,dfc1);matrix_free(a2pT);
            Matrix*w1T=matrix_transpose(fcw);Matrix*da2p=matrix_matmul(dfc1,w1T);matrix_free(w1T);
            for(size_t j=0;j<fcw->rows*fcw->cols;j++)fcw->data[j]-=lr*dw1->data[j];
            for(size_t c=0;c<fcb->cols;c++){double s=0;for(size_t r=0;r<dfc1->rows;r++)s+=dfc1->data[r*fcb->cols+c];fcb->data[c]-=lr*s;}
            matrix_free(dw1);
            /* Backward Pool2 */
            Matrix*da2r=maxpool2d_backward(p2,da2p);
            /* ReLU2 backward: a2r holds pre-ReLU values */
            for(size_t j=0;j<da2r->rows*da2r->cols;j++)if(a2r->data[j]<=0)da2r->data[j]=0;
            /* Backward Conv2 */
            Matrix*da1p=conv2d_backward(c2,da2r);conv2d_update_weights(c2,lr);
            /* Backward Pool1 */
            Matrix*da1r=maxpool2d_backward(p1,da1p);
            /* ReLU1 backward */
            for(size_t j=0;j<da1r->rows*da1r->cols;j++)if(a1r->data[j]<=0)da1r->data[j]=0;
            /* Backward Conv1 */
            Matrix*dX=conv2d_backward(c1,da1r);conv2d_update_weights(c1,lr);
            
            /* Cleanup */
            matrix_free(X);matrix_free(a1);matrix_free(a1r);matrix_free(a1p);
            matrix_free(a2);matrix_free(a2r);matrix_free(a2p);matrix_free(fc1);matrix_free(out);
            matrix_free(dout);matrix_free(dfc1);matrix_free(da2p);matrix_free(da2r);
            matrix_free(da1p);matrix_free(da1r);matrix_free(dX);
            free(lbls); batches++;
        }
        printf("Epoch %d: loss=%.4f\n",ep+1,loss_sum/batches); fflush(stdout);
    }
    
    for(int i=0;i<n;i++)matrix_free(s[i].img);free(s);
    conv2d_free(c1);maxpool2d_free(p1);conv2d_free(c2);maxpool2d_free(p2);
    matrix_free(fcw);matrix_free(fcb);matrix_free(fcw2);matrix_free(fcb2);
    P("done");
    return 0;
}
