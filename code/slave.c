#include <slave.h>
#include <math.h>
#include <simd.h>
#include <assert.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "args.h"
#include "util.h"
#include "dma_macros.h"
#define asm __asm


// #define LWPF_KERNELS K(total) K(dma1) K(dma2) K(dma3) K(comp) 
// #define LWPF_UNIT U(TEST) 
// #include "/home/export/online3/swmore/opensource/lwpf2/lwpf2.h"


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


void calculate_QKV_64_reuse_W(Arg_QK_t arg_){
  dma_init();
  Arg_QK tmp_arg;
  Arg_QK_t arg;
  arg = &tmp_arg;
  
  pe_get(arg_,arg,sizeof(Arg_QK));
  dma_syn();

  const int id = _MYID;
  const int B = arg->B;
  const int S = arg->S;
  const int D = arg->D;
  const int N = arg->N;
  const int PD = D/N;
  const int rows_per_core = D/64;
  const int start_pos = rows_per_core*id*D;
  const float* x = arg->x;
  const float* w = arg->w;
  const float* Q = arg->Q;
  const float* K = arg->K;
  const float* V = arg->V;
  const int b = arg->b;
  int flag = 0;
  int x_len;
  int num_head,pos_in_head;

  int loop = rows_per_core;
  int tmp_loop = 1;
  int rows_per_loop = 1;

  if(S+D<1500){
    loop = ceil(rows_per_core/2.0);
    tmp_loop = 2;
    rows_per_loop = 2;
  }
  float w_Q_slave[D*rows_per_loop];
  float w_K_slave[D*rows_per_loop];
  float w_V_slave[D*rows_per_loop];
  float part_res_Q[S*rows_per_loop];
  float part_res_K[S*rows_per_loop];
  float part_res_V[S*rows_per_loop];
  x_len = 15000-(D+S)*3*rows_per_loop;
  float x_slave[x_len];
  float Q_simd[4];
  float K_simd[4];
  float V_simd[4];
  int get_rows = floor(x_len/D);

  floatv4 Q_v4=0,K_v4=0,V_v4=0;
  floatv4 tmpv4_Q,tmpv4_K,tmpv4_V,tmpv4_x;
  
  volatile unsigned long get_reply;

  for(int l=0;l<loop;l++){

    if(l==loop-1&&rows_per_core%2!=0){
      tmp_loop = 1;
      
    }
    pe_get(w+start_pos+l*rows_per_loop*D,&w_Q_slave[0],sizeof(float)*D*tmp_loop);
    pe_get(w+D*D+start_pos+l*rows_per_loop*D,&w_K_slave[0],sizeof(float)*D*tmp_loop);
    pe_get(w+D*D*2+start_pos+l*rows_per_loop*D,&w_V_slave[0],sizeof(float)*D*tmp_loop);
    dma_syn();
    
      //read w
     
      
    for(int s = 0;s<S;s++)
    {
      
      if(s%get_rows==0){
        flag = 0;
        if(s+get_rows>=S){
          get_rows = S-s;
        }
        if(id==0){
          get_reply= 0;
          athread_get(BCAST_MODE, x+b*S*D+s*D,&x_slave[0],sizeof(float)*D*get_rows,&get_reply,0xff,0,0);
          while(get_reply!=1);
          dma_syn();  
        }
        athread_syn(ARRAY_SCOPE,0xffff); 
      }

      for(int r = 0;r<tmp_loop;r++)
      {  

        // floatv4 Q_v4=0,K_v4=0,V_v4=0;
        // floatv4 tmpv4_Q,tmpv4_K,tmpv4_V,tmpv4_x;
        Q_v4=0;
        K_v4=0;
        V_v4=0;

        for(int i=0;i<D;i+=4)
        {
          simd_load(tmpv4_x, &x_slave[i+D*flag]);
          simd_load(tmpv4_Q, &w_Q_slave[i+r*D]);
          simd_load(tmpv4_K, &w_K_slave[i+r*D]);
          simd_load(tmpv4_V, &w_V_slave[i+r*D]);

          Q_v4 = Q_v4+tmpv4_x*tmpv4_Q;
          K_v4 = K_v4+tmpv4_x*tmpv4_K;
          V_v4 = V_v4+tmpv4_x*tmpv4_V;
        }
        
        simd_store(Q_v4, Q_simd);
        part_res_Q[s+r*S] = Q_simd[0] + Q_simd[1] + Q_simd[2] + Q_simd[3];

        simd_store(K_v4, K_simd);
        part_res_K[s+r*S] = K_simd[0] + K_simd[1] + K_simd[2] + K_simd[3];
         

        simd_store(V_v4, V_simd);
        part_res_V[s+r*S] = V_simd[0] + V_simd[1] + V_simd[2] + V_simd[3];


      }
      flag++;
    }
    for(int r = 0;r<tmp_loop;r++){
      num_head = floor((r+id*rows_per_core+l*rows_per_loop)/PD);
      pos_in_head = (r+id*rows_per_core+l*rows_per_loop)%PD;
      pe_put(Q+b*S*D+num_head*S*PD+pos_in_head*S,&part_res_Q[r*S],sizeof(float)*S);
      pe_put(K+b*S*D+num_head*S*PD+pos_in_head*S,&part_res_K[r*S],sizeof(float)*S);
      pe_put(V+b*S*D+num_head*S*PD+pos_in_head*S,&part_res_V[r*S],sizeof(float)*S);
      dma_syn();
    }
    
  }
}







//需要更改
//极限情况，都不能拿到内存中,还有优化空间，比如一次尽可能多的放到LDM中
#define QK_normal_QN_size 1024
#define QK_normal_KN_size 5784
#define QK_normal_S 8192
//没有余数版本
void calculate_QK_normal(Arg_QK_t arg_){
  
  dma_init();
  

  Arg_QK tmp_arg;
  Arg_QK_t arg;
  arg = &tmp_arg;
  
  pe_get(arg_,arg,sizeof(Arg_QK));
  dma_syn();

  const int id = _MYID;
  const int B = arg->B;
  const int S = arg->S;
  const int D = arg->D;
  const int N = arg->N;
  volatile unsigned long get_reply;
  const int PD = D/N;
 
  //一个从核需要读取的行数
  int rows_per_core = floor(S/64);
  int rows_per_loop = rows_per_core/2;
  int loop = 2;
  //当前从核的开始位置
  int start_pos = id*rows_per_core*PD;
  //写回的开始列
  // int start_write = id*rows_per_core*S;
  const float* QN = arg->QN;
  const float* KN = arg->KN;
  const float* QK = arg->QK;
  const float* VN = arg->VN;
  const float* res = arg->y;
  float array_simd[4];
  


  // //临时数组
  float QN_slave[QK_normal_QN_size];
  float QK_slave[QK_normal_S];
  // int KN_len = 15000-PD-S;
  float KN_slave[QK_normal_KN_size];
  int get_kn = 90;
  int tmp_get_kn = 0;
  
  double sum = 0.0f;


  //phase3
  //*****************************************************
  int get_vn=5;
  int tmp_get_vn;
  int start_rows = rows_per_core*id;
  //*****************************************************

  floatv4 t1,t2,t3,t4,t5,t6,t7,t8;
  floatv4 tmp_resV41 = 0;
  floatv4 tmp_resV42 = 0;
  floatv4 tmp_resV43 = 0;
  floatv4 tmp_resV44 = 0;

  //假设计算哪行取哪一行

  
  for(int b=0;b<B;b++){
    for(int n=0;n<N;n++){

      for(int l = 0;l<loop;l++){
        pe_get(QN+b*S*D+n*S*PD+start_pos+l*rows_per_loop*PD,&QN_slave[0],sizeof(float)*PD*rows_per_loop);
        dma_syn();
        
        for(int ks = 0;ks<S;ks++){
          // athread_syn(ARRAY_SCOPE,0xffff);
          if(ks%get_kn==0){
            tmp_get_kn = get_kn;
            if(tmp_get_kn+ks>S){
              tmp_get_kn = S-ks;
            }
            if(id==0){
              get_reply= 0;
              athread_get(BCAST_MODE, KN+b*S*D+n*S*PD+ks*PD,&KN_slave[0],sizeof(float)*PD*tmp_get_kn,&get_reply,0xff,0,0);
              while(get_reply!=1);
              dma_syn();
            }
            athread_syn(ARRAY_SCOPE,0xffff);
          }
          

          for(int qs = 0;qs<rows_per_loop;qs++){

        

            // floatv4 t1,t2,t3,t4 ;
            // floatv4 tmp_resV41 = 0;
            // floatv4 tmp_resV42 = 0;
            tmp_resV41 = 0;
            tmp_resV42 = 0;
           
            
            for(int i=0;i<PD;i+=8){
              // tmp+=QN_slave[qs*PD+i]*KN_slave[i+ks*PD];
              simd_load(t1,&QN_slave[i+qs*PD]);
              simd_load(t2,&KN_slave[i+(ks%get_kn)*PD]);


              simd_load(t3,&QN_slave[i+4+qs*PD]);
              simd_load(t4,&KN_slave[i+4+(ks%get_kn)*PD]);

              tmp_resV41 += t1*t2;
              tmp_resV42 += t3*t4;
            }
            tmp_resV41 +=tmp_resV42;
            simd_store(tmp_resV41,array_simd);
            QK_slave[ks+qs*S]=(array_simd[0]+array_simd[1])+(array_simd[2]+array_simd[3]);
            

          }

        }

        for(int r = 0;r<rows_per_loop;r++){
          sum = 0;
          for(int i = r*S;i<r*S+S;i++){
            sum+=QK_slave[i];
          }

          for(int i = r*S;i<r*S+S;i++){
            QK_slave[i] = QK_slave[i]*(1/sum);
          }
        }
        // pe_put(QK+b*S*S*N+n*S*S+start_write+l*rows_per_loop*S,&QK_slave[0],sizeof(float)*S*rows_per_loop);
        // dma_syn();

        for(int vn_col = 0;vn_col<PD;vn_col++){
          if(vn_col%get_vn==0){
            tmp_get_vn = get_vn;
            if(tmp_get_vn+vn_col>PD){
              tmp_get_vn = PD- vn_col;
            }
            if(id==0){
              get_reply= 0;
              athread_get(BCAST_MODE, VN+b*S*D+n*S*PD+vn_col*S,&KN_slave[0],sizeof(float)*S*tmp_get_vn,&get_reply,0xff,0,0);
              while(get_reply!=1);
              dma_syn();
            }
            athread_syn(ARRAY_SCOPE,0xffff);
          }

          for(int qk_r = 0;qk_r<rows_per_loop;qk_r++){
            // floatv4 t1,t2,t3,t4,t5,t6,t7,t8;
            // floatv4 tmp_resV41 = 0;
            // floatv4 tmp_resV42 = 0;
            // floatv4 tmp_resV43 = 0;
            // floatv4 tmp_resV44 = 0;

            tmp_resV41 = 0;
            tmp_resV42 = 0;
            tmp_resV43 = 0;
            tmp_resV44 = 0;

            for(int i=0;i<S;i+=16){
        
              // tmp+=QK_slave[i]*VN_slave[i];
              simd_load(t1,&QK_slave[i+qk_r*S]);
              simd_load(t2,&KN_slave[i+(vn_col%get_vn)*S]);

              simd_load(t3,&QK_slave[i+4+qk_r*S]);
              simd_load(t4,&KN_slave[i+4+(vn_col%get_vn)*S]);

              simd_load(t5,&QK_slave[i+8+qk_r*S]);
              simd_load(t6,&KN_slave[i+8+(vn_col%get_vn)*S]);

              simd_load(t7,&QK_slave[i+12+qk_r*S]);
              simd_load(t8,&KN_slave[i+12+(vn_col%get_vn)*S]);

              tmp_resV41 += t1*t2;
              tmp_resV42 += t3*t4;
              tmp_resV43 += t5*t6;
              tmp_resV44 += t7*t8;
              
            }
            tmp_resV41+=tmp_resV42;
            tmp_resV43+=tmp_resV44;
            tmp_resV41+=tmp_resV43;
            simd_store(tmp_resV41,array_simd);
            QN_slave[vn_col+qk_r*PD]=(array_simd[0]+array_simd[1])+(array_simd[2]+array_simd[3]);
          }
        }
        
        get_reply = 0;
        athread_put(PE_MODE,&QN_slave[0],res+b*S*D+n*PD+(start_rows+l*rows_per_loop)*D,sizeof(float)*PD*rows_per_loop,&get_reply,sizeof(float)*(D-PD),sizeof(float)*PD);
        while(get_reply!=1);
        dma_syn(); 

      }

    }
  }
  

}

// #define QK_little_fast_QN_size 2048
// #define QK_little_fast_KN_size 8856
// #define QK_little_fast_S 4096

void calculate_QK_little_fast(Arg_QK_t arg_){
  volatile unsigned long get_reply;
  dma_init();


  Arg_QK tmp_arg;
  Arg_QK_t arg;
  arg = &tmp_arg;
  
  // pe_get(arg_,arg,sizeof(Arg_QK));
  get_reply = 0;
  athread_get(PE_MODE,arg_,arg,sizeof(float)*16,&get_reply,0,0,0);
  while(get_reply!=1);
  dma_syn();

  const int id = _MYID;
  const int B = arg->B;
  const int S = arg->S;
  const int D = arg->D;
  const int N = arg->N;
  
  const int PD = D/N;
  
 
  //一个从核需要读取的行数
  int rows_per_core = floor(S/64);
  //当前从核的开始位置
  int start_pos = id*rows_per_core*PD;
  //写回的开始列
  // int start_write = id*rows_per_core*S;
  const float* QN = arg->QN;
  const float* KN = arg->KN;
  const float* QK = arg->QK;
  const float* VN = arg->VN;
  float* res = arg->y;
  float array_simd[4];
  
  int tmp_get_kn = 0;

  //临时数组
  int QK_len = floor(S*S/64)+S;
  int KN_len = 14500-QK_len-1536;
  float QN_slave[1536];
  float KN_slave[KN_len];
  float QK_slave[QK_len];
  int get_kn = floor(KN_len/PD);
  if(get_kn>S){
    get_kn = S;
  }
  
  double sum = 0.0f;




//phase3
//************************************************
 
  //写回的开始列
  int start_write = id*rows_per_core;
  int get_vn=floor(KN_len/S);
  if(get_vn>PD){
    get_vn = PD;
  }
  int tmp_get_vn;
//*************************************************
  int mod = S%64;
  //如果存在余数
  if(mod!=0){
    if(id<mod){
      rows_per_core+=1;
      start_pos = id*rows_per_core*PD;
      
      start_write = id*rows_per_core;
   }
    else{
      start_pos = (id*rows_per_core+mod)*PD;
      
      start_write = (id*rows_per_core+mod);
    }
  }

  int size1 = PD*rows_per_core;
  int tmp = S;


  floatv4 t1,t2,t3,t4,t5,t6,t7,t8;
  floatv4 tmp_resV41 = 0;
  floatv4 tmp_resV42 = 0;
  floatv4 tmp_resV43 = 0;
  floatv4 tmp_resV44 = 0;
  

  //假设计算哪行取哪一行

  
  for(int b=0;b<B;b++){
    for(int n=0;n<N;n++){

      // pe_get(QN+b*S*D+n*S*PD+start_pos,&QN_slave[0],sizeof(float)*PD*rows_per_core);
      get_reply = 0;
      athread_get(PE_MODE,QN+b*S*D+n*S*PD+start_pos,&QN_slave[0],sizeof(float)*size1,&get_reply,0,0,0);
      while(get_reply!=1);
      dma_syn();
      
      for(int ks = 0;ks<S;ks++){
        // athread_syn(ARRAY_SCOPE,0xffff);
        if(ks%get_kn==0){
          tmp_get_kn = get_kn;
          if(tmp_get_kn+ks>S){
            tmp_get_kn = S-ks;
          }
          if(id==0){
            get_reply= 0;
            athread_get(BCAST_MODE, KN+b*S*D+n*S*PD+ks*PD,&KN_slave[0],sizeof(float)*PD*tmp_get_kn,&get_reply,0xff,0,0);
            while(get_reply!=1);
            dma_syn();
          }
          athread_syn(ARRAY_SCOPE,0xffff);
        }
        

        for(int qs = 0;qs<rows_per_core;qs++){

      

         
          // floatv4 t1,t2,t3,t4 ;
          // floatv4 tmp_resV41 = 0;
          // floatv4 tmp_resV42 = 0;
          tmp_resV41 = 0;
          tmp_resV42 = 0;
            
            
          for(int i=0;i<PD;i+=8){
            // tmp+=QN_slave[qs*PD+i]*KN_slave[i+ks*PD];
            simd_load(t1,&QN_slave[i+qs*PD]);
            simd_load(t2,&KN_slave[i+(ks%get_kn)*PD]);


            simd_load(t3,&QN_slave[i+4+qs*PD]);
            simd_load(t4,&KN_slave[i+4+(ks%get_kn)*PD]);

            tmp_resV41 += t1*t2;
            tmp_resV42 += t3*t4;
          }
          tmp_resV41 +=tmp_resV42;
          simd_store(tmp_resV41,array_simd);
          QK_slave[ks+qs*S]=(array_simd[0]+array_simd[1])+(array_simd[2]+array_simd[3]);
          

        }

      }

      for(int r = 0;r<rows_per_core;r++){
        sum = 0;
        for(int i = r*S;i<r*S+S;i++){
          sum+=QK_slave[i];
        }

        for(int i = r*S;i<r*S+S;i++){
          QK_slave[i] = QK_slave[i]*(1/sum);
        }
      }
      // pe_put(QK+b*S*S*N+n*S*S+start_write,&QK_slave[0],sizeof(float)*S*rows_per_core);
      // dma_syn();
      for(int vn_col = 0;vn_col<PD;vn_col++){
        if(vn_col%get_vn==0){
          tmp_get_vn = get_vn;
          if(tmp_get_vn+vn_col>PD){
            tmp_get_vn = PD- vn_col;
          }
          if(id==0){
            get_reply= 0;
            athread_get(BCAST_MODE, VN+b*S*D+n*S*PD+vn_col*S,&KN_slave[0],sizeof(float)*S*tmp_get_vn,&get_reply,0xff,0,0);
            while(get_reply!=1);
            dma_syn();
          }
          athread_syn(ARRAY_SCOPE,0xffff);
        }

        for(int qk_r = 0;qk_r<rows_per_core;qk_r++){
          tmp_resV41 = 0;
          tmp_resV42 = 0;
          tmp_resV43 = 0;
          tmp_resV44 = 0;


          if(S%16!=0){
            tmp = S-(S%16);
          }

          for(int i=0;i<tmp;i+=16){
      
            // tmp+=QK_slave[i]*VN_slave[i];
            simd_load(t1,&QK_slave[i+qk_r*S]);
            simd_load(t2,&KN_slave[i+(vn_col%get_vn)*S]);

            simd_load(t3,&QK_slave[i+4+qk_r*S]);
            simd_load(t4,&KN_slave[i+4+(vn_col%get_vn)*S]);

            simd_load(t5,&QK_slave[i+8+qk_r*S]);
            simd_load(t6,&KN_slave[i+8+(vn_col%get_vn)*S]);

            simd_load(t7,&QK_slave[i+12+qk_r*S]);
            simd_load(t8,&KN_slave[i+12+(vn_col%get_vn)*S]);

            tmp_resV41 += t1*t2;
            tmp_resV42 += t3*t4;
            tmp_resV43 += t5*t6;
            tmp_resV44 += t7*t8;
            
          }
          tmp_resV41+=tmp_resV42;
          tmp_resV43+=tmp_resV44;
          tmp_resV41+=tmp_resV43;
          simd_store(tmp_resV41,array_simd);
          QN_slave[vn_col+qk_r*PD]=(array_simd[0]+array_simd[1])+(array_simd[2]+array_simd[3]);
          if(S%16!=0){
            for(int i=tmp;i<S;i++){
              QN_slave[vn_col+qk_r*PD]+=QK_slave[i+qk_r*S]*KN_slave[i+(vn_col%get_vn)*S];
            }
          }
        }
      }
      
      get_reply = 0;
      athread_put(PE_MODE,&QN_slave[0],res+b*S*D+n*PD+(start_write)*D,sizeof(float)*PD*rows_per_core,&get_reply,sizeof(float)*(D-PD),sizeof(float)*PD);
      while(get_reply!=1);
      dma_syn(); 


    }

  }
  

}


// #define QK_fast_QN_size 4096//128
// #define QK_fast_KN_size 8192
// #define QK_fast_S 256

// //之前是划分的KN，现在划分QN
// void calculate_QK_fast(Arg_QK_t arg_){
//   dma_init();

//   Arg_QK tmp_arg;
//   Arg_QK_t arg;
//   arg = &tmp_arg;
  
//   pe_get(arg_,arg,sizeof(Args));
//   dma_syn();

//   const int id = _MYID;
//   const int B = arg->B;
//   const int S = arg->S;
//   const int D = arg->D;
//   const int N = arg->N;
//   // const int n = arg->n;
//   // const int b= arg->b;
//   volatile unsigned long get_reply;
//   const int PD = D/N;
 
//   //一个从核需要读取的行数
//   int rows_per_core = floor(S/64);
//   //当前从核的开始位置
//   int start_pos = id*rows_per_core*PD;
//   //写回的开始列
//   int start_write = id*rows_per_core*S;
//   const float* QN = arg->QN;
//   const float* KN = arg->KN;
//   const float* QK = arg->QK;
//   float array_simd[4];
//   //临时数组
//   float KN_slave[QK_fast_KN_size];
//   float QN_slave[QK_fast_QN_size];
//   float QK_slave[QK_fast_S];
  
//   double sum = 0.0f;

  
//   int mod = S%64;
//   //如果存在余数,那么有的从核将多计算一行
//   if(mod!=0){
//     if(id<mod){
//       rows_per_core+=1;
//       start_pos = id*rows_per_core*PD;
//       start_write = id*rows_per_core*S;
//    }
//     else{
//       start_pos = (id*rows_per_core+mod)*PD;
//       start_write = (id*rows_per_core+mod)*S;
//     }
//   }
  
//   //试图直接进行归一化处理
  

//   //假设都能拿进来。不考虑其他因素
//   //KN
//   for(int b=0;b<B;b++){
//     for(int n = 0;n<N;n++){
      
      
//       if(id==0){
//         get_reply= 0;
//         athread_get(BCAST_MODE, KN+b*S*D+n*S*PD,&KN_slave[0],sizeof(float)*PD*S,&get_reply,0xff,0,0);
//         while(get_reply!=1);
//         dma_syn();
//       }
//       athread_syn(ARRAY_SCOPE,0xffff);

//       // pe_get(QN+b*S*D+n*S*PD,&QN_slave[0],sizeof(float)*PD*S);
//       pe_get(QN+b*S*D+n*S*PD+start_pos,&QN_slave[0],sizeof(float)*PD*rows_per_core);
//       dma_syn();
      
//       for(int qs = 0;qs<rows_per_core;qs++){
//         sum = 0;
//         for(int ks=0;ks<S;ks++){
          
          

//           floatv4 t1,t2;
//           floatv4 tmp_resV4 = 0;
          
//           //PD是32的倍数，使用SIMD
//           for(int i=0;i<PD;i+=4){
//             // tmp+=QN_slave[qs*PD+i]*KN_slave[i+ks*PD];
//             simd_load(t1,&QN_slave[qs*PD+i]);
//             simd_load(t2,&KN_slave[i+ks*PD]);

//             tmp_resV4 += t1*t2;
//           }
//           simd_store(tmp_resV4,array_simd);
//           QK_slave[qs*S+ks]=array_simd[0]+array_simd[1]+array_simd[2]+array_simd[3];
         
          
//           sum+=QK_slave[qs*S+ks];
          
          
//         }

//         for(int i=qs*S;i<(qs*S+S);i++){
//           QK_slave[i] = QK_slave[i]*(1/sum);
//         }
//       }

//       //写回是最占时间的
      
//       get_reply = 0;
//       athread_put(PE_MODE,&QK_slave[0],QK+b*S*S*N+n*S*S+start_write,sizeof(float)*rows_per_core*S,&get_reply,0,0);
//       while(get_reply!=1);
//       dma_syn();
      
//     }
//   }
  
// }

// #define QK_size 625
// #define VN_size 12800//200*64
// #define res_size 128
// void calculate_QK_V(Arg_QK_t arg_){
//   dma_init();

//   Arg_QK tmp_arg;
//   Arg_QK_t arg;
//   arg = &tmp_arg;
  
//   pe_get(arg_,arg,sizeof(Args));
//   dma_syn();

//   const int id = _MYID;
//   const int B = arg->B;
//   const int S = arg->S;
//   const int D = arg->D;
//   const int N = arg->N;
//   const int PD = D/N;
//   volatile unsigned long get_reply;
//   //对QK进行划分
//   int rows_per_core = floor(S/64);
//   int start_pos = rows_per_core*id*S;
//   int start_rows = rows_per_core*id;
//   float* QK = arg->QK;
//   float* VN = arg->VN;
//   float* res = arg->y;
//   float array_simd[4];
//   float QK_slave[QK_size];
//   float VN_slave[VN_size];
//   float res_slave[res_size];
//   float tmp;

//   int mod = S%64;
//   if(mod!=0){
//     if(id<mod){
//       rows_per_core+=1;
//       start_pos = id*rows_per_core*S;
//       start_rows = id*rows_per_core;
//     }
//     else{
//       start_pos = (id*rows_per_core+mod)*S;
//       start_rows = id*rows_per_core+mod;
//     }
//   } 

  
//   //get data
//   for(int b = 0;b<B;b++){

//     for(int n=0;n<N;n++){
//       // athread_syn(ARRAY_SCOPE,0xffff);
//       if(id==0){
//         get_reply= 0;
//         athread_get(BCAST_MODE, VN+b*S*D+n*S*PD,&VN_slave[0],sizeof(float)*PD*S,&get_reply,0xff,0,0);
//         while(get_reply!=1);
//         dma_syn();
//       }
//       athread_syn(ARRAY_SCOPE,0xffff);
      

//       pe_get(QK+b*S*S*N+n*S*S+start_pos,&QK_slave[0],sizeof(float)*S*rows_per_core);
//       dma_syn();

      
      

      
//       for(int qk_r = 0;qk_r<rows_per_core;qk_r++){
//         for(int vn_col = 0;vn_col<PD;vn_col++){
//           tmp = 0;

//           floatv4 t1,t2;
//           floatv4 tmp_resV4 = 0;
//           for(int i=0;i<S;i+=4){
      
//             // tmp+=QK_slave[i]*VN_slave[i];
//             simd_load(t1,&QK_slave[qk_r*S+i]);
//             simd_load(t2,&VN_slave[i+vn_col*S]);

//             tmp_resV4 += t1*t2;
            
//           }
//           simd_store(tmp_resV4,array_simd);
//           tmp=array_simd[0]+array_simd[1]+array_simd[2]+array_simd[3];
//           res_slave[vn_col] = tmp;
          
//         }
        
//         pe_put(res+b*S*D+n*PD+(start_rows+qk_r)*D,&res_slave[0],sizeof(float)*PD);
//         dma_syn(); 
        
//       }

//     }
//   }
// }


















