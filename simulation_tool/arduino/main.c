//
// Created by czerwins on 2017-05-15.
//
#include <stdio.h>
#include <stdlib.h>

#define STATES_NUM 72
#define ACTION_NUM 4
#define initial_LEARNING_RATE 1
#define initial_DISCOUNT_FACTOR 0.5
#define initial_E_GREEDY 0.9

int j, i, action, reward;
float max;

void initialize(float (*q_tab)[ACTION_NUM], float *lr, float *df, float *eps){
    for( i = 0; i < STATES_NUM ; i++){
        for( j = 0; j < ACTION_NUM ; j++){
            q_tab[i][j] = 0;
        }
    }
    *lr = initial_LEARNING_RATE;
    *df = initial_DISCOUNT_FACTOR;
    *eps = initial_E_GREEDY;
}

void update_knowledge(float (*q_tab)[ACTION_NUM], int reward, int next_state, int state, float learning_rate,
                      float discount_factor, int action){
    j = 0;
    max = q_tab[next_state][0];
    for(i = 1; i < ACTION_NUM; i++){
        if(q_tab[next_state][i] > max){
            max = q_tab[next_state][i];
            j = i;
        }
    }
    float delta = learning_rate * ( ((float)reward) + discount_factor * max - q_tab[state][action]);
    q_tab[state][action] += delta;
}

int choose_action(float (*q_tab)[ACTION_NUM], int state, float eps){
    if(((double)rand())/RAND_MAX > 1.0 - eps){  // random choose
        return rand() % ACTION_NUM;
    }else{                                      // arg max
        j = 0;
        max = q_tab[state][0];
        for(i = 1; i < ACTION_NUM; i++){
            if(q_tab[state][i] > max){
                max = q_tab[state][i];
                j = i;
            }
        }
        return j;
    }
}

int main(void){
    float Q[STATES_NUM][ACTION_NUM];
//    int actions[] = {60, 45, 35, 30, 16, 8, 4, 2, 1};
    float discount_factor, learning_rate, eps;
    int prev_state = -1;
    printf("%f;%f;%f\n", learning_rate,discount_factor,eps);
    initialize(Q, &learning_rate, &discount_factor, &eps);
    printf("%f;%f;%f\n", learning_rate,discount_factor,eps);
    int i,j;
    for(i = 0; i < 50; i++){
        for(j = 0; j < STATES_NUM; j++) {
            if ( prev_state >= 0) update_knowledge(Q, reward, j, prev_state, learning_rate, discount_factor, action);
            action = choose_action(Q, j, eps);
            reward = (rand() % 100) - 50;
            prev_state = j;
        }
    }
    for( i = 0; i < STATES_NUM ; i++){
        for( j = 0; j < ACTION_NUM ; j++){
            printf("%.2f;",Q[i][j]);
        }
        printf("\n");
    }
    return 0;
}