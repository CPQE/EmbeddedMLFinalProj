#pragma once
#include <stdint.h>

#define SAMPLE_COUNT 2
#define SAMPLE_TIME   600
#define SAMPLE_FEATS  23

extern const float samples_X[SAMPLE_COUNT][SAMPLE_TIME][SAMPLE_FEATS];
extern const int   samples_y[SAMPLE_COUNT];
extern const float scaler_mean[SAMPLE_FEATS];
extern const float scaler_std[SAMPLE_FEATS];
