/*************************************************************************
	> File Name: convolution_forward.c
	> Author: 
	> Mail: 
	> Created Time: 
 ************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <athread.h>
#include <math.h>

#include "args.h"
#include "util.h"


// #define MPE
// #define LWPF_UNITS U(TEST)
// #include "/home/export/online3/swmore/opensource/lwpf2/lwpf2.h"



extern void SLAVE_FUN(calculate_QKV_64_reuse_W)();


extern void SLAVE_FUN(calculate_QK_normal)();
extern void SLAVE_FUN(calculate_QK_little_fast)();




typedef struct Arg_QK {
    int B; // batch
    int S; // sequence length
    int D; // vector size
    int N; // number of heads
    int b;
    int n;
    float* y; 
    float* QN;
    float* KN;
    float* QK;
    float* VN;
    float* x; // input x
    float* w; // Wq, Wk, Wv
    float* Q; 
    float* K;
    float* V;

} Arg_QK, *Arg_QK_t;

static inline void trans_matrix(float* matrix,float* res, int B,int N,int PD,int S){
    for(int b = 0;b<B;b++)
        for(int n = 0;n<N;n++)
        {
            int tmp = b*S*PD*N+n*S*PD;
            for(int r = 0;r<PD;r++)
                for(int c = 0;c<S;c++){
                    res[tmp+c*PD+r] = matrix[tmp+r*S+c]; 
                }
        }
            

}

static inline void trans_matrix_b(float* matrix,float* res, int B,int N,int PD,int S, int b){
    
    for(int n = 0;n<N;n++)
    {
        int tmp = b*S*PD*N+n*S*PD;
        for(int r = 0;r<PD;r++)
            for(int c = 0;c<S;c++){
                res[tmp+c*PD+r] = matrix[tmp+r*S+c]; 
            }
    }
            

}



int multihead_attention(Args_t arg)
{   

	//TIME_T t1, t2;

    //MARK_TIME(t1);
    //相关参数和数据
		const int B = arg->B;
    const int S = arg->S;
    const int D = arg->D;
    const int N = arg->N;
    float* Q = arg->Q;
    float* K = arg->K;
    float* V = arg->V;
    float* QK = arg->QK;
    // float* y = arg->y;
		const int PD = D/N;
    float* VN = V;
    float* QN = (float*)aligned_malloc(sizeof(float)*B*S*D, 128);
    float* KN = (float*)aligned_malloc(sizeof(float)*B*S*D, 128);

   
    Arg_QK_t arg_qk = (Arg_QK_t)malloc(sizeof(Arg_QK));
    arg_qk->B = B;
    arg_qk->S = S;
    arg_qk->D = D;
    arg_qk->N = N;
    arg_qk->QN = QN;
    arg_qk->KN = KN;
    arg_qk->QK = QK;
    arg_qk->VN = VN;
    arg_qk->y = arg->y;
    arg_qk->x = arg->x;
    arg_qk->w = arg->w;
    arg_qk->Q = Q;
    arg_qk->K = K;
    arg_qk->V = V;


    
    if(B==1){
        arg_qk->b = 0;
        athread_spawn(calculate_QKV_64_reuse_W,arg_qk);
        athread_join(); // wait for all slave threads finished
        trans_matrix(Q,QN, B,N,PD,S);
        trans_matrix(K,KN, B,N,PD,S);
    }
    else{

        arg_qk->b = 0;
        athread_spawn(calculate_QKV_64_reuse_W,arg_qk);
        athread_join(); // wait for all slave threads finished
        for(int b=1;b<B;b++){
            arg_qk->b = b;
            athread_spawn(calculate_QKV_64_reuse_W,arg_qk);
            trans_matrix_b(Q,QN, B,N,PD,S,b-1);
            trans_matrix_b(K,KN, B,N,PD,S,b-1);
            athread_join(); // wait for all slave threads finished

        }

        trans_matrix_b(Q,QN, B,N,PD,S,B-1);
        trans_matrix_b(K,KN, B,N,PD,S,B-1);
    }

    
        

   

   // MARK_TIME(t2);
    //LOG("T1 time : %.3f ms", DIFF_TIME(t1, t2));

    



    //MARK_TIME(t1);
    //LOG("trans time : %.3f ms", DIFF_TIME(t2, t1));

    
    
		//}
     if(S<=768){

        athread_spawn(calculate_QK_little_fast,arg_qk);
        athread_join();
    }
    else{ //一般情况
        athread_spawn(calculate_QK_normal,arg_qk);
        athread_join();
           
    }

    //test
    
    //MARK_TIME(t2);
    //LOG("T2 time : %.3f ms", DIFF_TIME(t1, t2));


    aligned_free(QN);
    aligned_free(KN);
    
    // lwpf_report_summary(stdout, &conf);
    return 0;
}

