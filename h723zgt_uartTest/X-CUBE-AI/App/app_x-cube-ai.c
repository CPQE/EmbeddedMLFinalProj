
/**
  ******************************************************************************
  * @file    app_x-cube-ai.c
  * @author  X-CUBE-AI C code generator
  * @brief   AI program body
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2026 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  *
  ******************************************************************************
  */

 /*
  * Description
  *   v1.0 - Minimum template to show how to use the Embedded Client API
  *          model. Only one input and one output is supported. All
  *          memory resources are allocated statically (AI_NETWORK_XX, defines
  *          are used).
  *          Re-target of the printf function is out-of-scope.
  *   v2.0 - add multiple IO and/or multiple heap support
  *
  *   For more information, see the embeded documentation:
  *
  *       [1] %X_CUBE_AI_DIR%/Documentation/index.html
  *
  *   X_CUBE_AI_DIR indicates the location where the X-CUBE-AI pack is installed
  *   typical : C:\Users\[user_name]\STM32Cube\Repository\STMicroelectronics\X-CUBE-AI\7.1.0
  */

#ifdef __cplusplus
 extern "C" {
#endif

/* Includes ------------------------------------------------------------------*/

#if defined ( __ICCARM__ )
#elif defined ( __CC_ARM ) || ( __GNUC__ )
#endif

/* System headers */
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <inttypes.h>
#include <string.h>

#include "app_x-cube-ai.h"
#include "main.h"
#include "ai_datatypes_defines.h"
#include "road_cnn.h"
#include "road_cnn_data.h"

/* USER CODE BEGIN includes */
//#include "samples_fp16.h"
#include "samples_fp32.h"
#include "i2c_lcd.h"
/* USER CODE END includes */

/* IO buffers ----------------------------------------------------------------*/

#if !defined(AI_ROAD_CNN_INPUTS_IN_ACTIVATIONS)
AI_ALIGNED(4) ai_i8 data_in_1[AI_ROAD_CNN_IN_1_SIZE_BYTES];
ai_i8* data_ins[AI_ROAD_CNN_IN_NUM] = {
data_in_1
};
#else
ai_i8* data_ins[AI_ROAD_CNN_IN_NUM] = {
NULL
};
#endif

#if !defined(AI_ROAD_CNN_OUTPUTS_IN_ACTIVATIONS)
AI_ALIGNED(4) ai_i8 data_out_1[AI_ROAD_CNN_OUT_1_SIZE_BYTES];
ai_i8* data_outs[AI_ROAD_CNN_OUT_NUM] = {
data_out_1
};
#else
ai_i8* data_outs[AI_ROAD_CNN_OUT_NUM] = {
NULL
};
#endif

/* Activations buffers -------------------------------------------------------*/

AI_ALIGNED(32)
static uint8_t pool0[AI_ROAD_CNN_DATA_ACTIVATION_1_SIZE];

ai_handle data_activations0[] = {pool0};

/* AI objects ----------------------------------------------------------------*/

static ai_handle road_cnn = AI_HANDLE_NULL;

static ai_buffer* ai_input;
static ai_buffer* ai_output;

static void ai_log_err(const ai_error err, const char *fct)
{
  /* USER CODE BEGIN log */
  if (fct)
    printf("TEMPLATE - Error (%s) - type=0x%02x code=0x%02x\r\n", fct,
        err.type, err.code);
  else
    printf("TEMPLATE - Error - type=0x%02x code=0x%02x\r\n", err.type, err.code);

  do {} while (1);
  /* USER CODE END log */
}

static int ai_boostrap(ai_handle *act_addr)
{
  ai_error err;

  /* Create and initialize an instance of the model */
  err = ai_road_cnn_create_and_init(&road_cnn, act_addr, NULL);
  if (err.type != AI_ERROR_NONE) {
    ai_log_err(err, "ai_road_cnn_create_and_init");
    return -1;
  }

  ai_input = ai_road_cnn_inputs_get(road_cnn, NULL);
  ai_output = ai_road_cnn_outputs_get(road_cnn, NULL);

#if defined(AI_ROAD_CNN_INPUTS_IN_ACTIVATIONS)
  /*  In the case where "--allocate-inputs" option is used, memory buffer can be
   *  used from the activations buffer. This is not mandatory.
   */
  for (int idx=0; idx < AI_ROAD_CNN_IN_NUM; idx++) {
	data_ins[idx] = ai_input[idx].data;
  }
#else
  for (int idx=0; idx < AI_ROAD_CNN_IN_NUM; idx++) {
	  ai_input[idx].data = data_ins[idx];
  }
#endif

#if defined(AI_ROAD_CNN_OUTPUTS_IN_ACTIVATIONS)
  /*  In the case where "--allocate-outputs" option is used, memory buffer can be
   *  used from the activations buffer. This is no mandatory.
   */
  for (int idx=0; idx < AI_ROAD_CNN_OUT_NUM; idx++) {
	data_outs[idx] = ai_output[idx].data;
  }
#else
  for (int idx=0; idx < AI_ROAD_CNN_OUT_NUM; idx++) {
	ai_output[idx].data = data_outs[idx];
  }
#endif

  return 0;
}

static int ai_run(void)
{
  ai_i32 batch;

  batch = ai_road_cnn_run(road_cnn, ai_input, ai_output);
  if (batch != 1) {
    ai_log_err(ai_road_cnn_get_error(road_cnn),
        "ai_road_cnn_run");
    return -1;
  }

  return 0;
}

/* USER CODE BEGIN 2 */
extern UART_HandleTypeDef huart3;
extern I2C_HandleTypeDef hi2c1;
static int last_class = -1;
static float fp16_to_float(uint16_t h)
{
    // use manual bit conversion (CMSIS or approximate)
    union { uint16_t u; float f; } conv;
    memcpy(&conv.u, &h, sizeof(uint16_t));
    return conv.f; // compiler may warn if strict FP16 not enabled
}

int acquire_and_process_data(ai_i8* data[], int sample_idx)
{
	float* input = (float*)data[0];  // point to Cube-AI input buffer
	for (int t = 0; t < SAMPLE_TIME; t++) {
		for (int f = 0; f < SAMPLE_FEATS; f++) {
			int idx_flat = t * SAMPLE_FEATS + f;
			input[idx_flat] = (samples_X[sample_idx][t][f] - scaler_mean[f]) / scaler_std[f];
		}
	}
	return 0;
}

int post_process(ai_i8* data[])
{
    float prob = ((float*)data[0])[0];
    int predicted_class = (prob > 1e-20f) ? 1 : 0;
    char buf[96];
    int len;
    if (predicted_class == 1){
    	len = snprintf(buf, sizeof(buf), "ATTACK! Prob: %.9e\r\n", prob);
    }
    else{
    	len = snprintf(buf, sizeof(buf), "Normal\ Prob: %.9e\r\n", prob);
    }
    HAL_UART_Transmit(&huart3, (uint8_t*)buf, len, HAL_MAX_DELAY);

    if (predicted_class != last_class) {
    	HAL_Delay(500);
        last_class = predicted_class;
        lcd_clear();
        lcd_put_cursor(0, 0);
        lcd_send_string(predicted_class ? "ATTACK!" : "Normal");
    }


    return 0;

}

/* USER CODE END 2 */

/* Entry points --------------------------------------------------------------*/

void MX_X_CUBE_AI_Init(void)
{
    /* USER CODE BEGIN 5 */
  printf("\r\nTEMPLATE - initialization\r\n");

  ai_boostrap(data_activations0);
    /* USER CODE END 5 */
}

void MX_X_CUBE_AI_Process(void)
{
    /* USER CODE BEGIN 6 */
    int res = 0;
    int sample_idx = 0;
    char buf[64];
//    HAL_UART_Transmit(&huart3, (uint8_t*)"TEMPLATE - run - main loop\r\n", 28, HAL_MAX_DELAY);
    if (road_cnn) {
        for (sample_idx = 0; sample_idx < SAMPLE_COUNT; sample_idx++) {
            /* 1 - acquire and pre-process input data for this sample */
            res = acquire_and_process_data(data_ins, sample_idx);
            if (res != 0) break;
            /* 2 - process the data - call inference engine */
            res = ai_run();
            //debugging
//            float *out = (float*)data_outs[0];
//            for (int i = 0; i < AI_ROAD_CNN_OUT_1_SIZE; i++) {
//            	printf("raw_out[0] = %.9e\r\n", out[0]);
//            }
             //debugging end
            if (res != 0) break;
            res = post_process(data_outs); /* post-process the predictions */
            if (res != 0) break;
        }
    }

    if (res != 0) {
        ai_error err = {AI_ERROR_INVALID_STATE, AI_ERROR_CODE_NETWORK};
        ai_log_err(err, "Process has FAILED");
        HAL_UART_Transmit(&huart3, (uint8_t*)"Process has FAILED\r\n", 21, HAL_MAX_DELAY);
    }
    /* USER CODE END 6 */
}
#ifdef __cplusplus
}
#endif
