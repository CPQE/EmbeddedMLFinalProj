#pragma once

#define SAMPLE_COUNT 4
#define SAMPLE_TIME   600
#define SAMPLE_FEATS  23
#define SAMPLE_SCALE  1000.0f

#include <stdint.h>

extern const int16_t samples_X[SAMPLE_COUNT][SAMPLE_TIME][SAMPLE_FEATS];
extern const int samples_y[SAMPLE_COUNT];

