
/**
  ******************************************************************************
  * @file    app_x-cube-ai.c
  * @author  X-CUBE-AI C code generator
  * @brief   AI program body
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2025 STMicroelectronics.
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
#include "road32.h"
#include "road32_data.h"

/* USER CODE BEGIN includes */
#include "samples.h"

static int current_sample_idx = 0;

extern UART_HandleTypeDef huart2;

/* USER CODE END includes */

/* IO buffers ----------------------------------------------------------------*/

#if !defined(AI_ROAD32_INPUTS_IN_ACTIVATIONS)
AI_ALIGNED(4) ai_i8 data_in_1[AI_ROAD32_IN_1_SIZE_BYTES];
ai_i8* data_ins[AI_ROAD32_IN_NUM] = {
data_in_1
};
#else
ai_i8* data_ins[AI_ROAD32_IN_NUM] = {
NULL
};
#endif

#if !defined(AI_ROAD32_OUTPUTS_IN_ACTIVATIONS)
AI_ALIGNED(4) ai_i8 data_out_1[AI_ROAD32_OUT_1_SIZE_BYTES];
ai_i8* data_outs[AI_ROAD32_OUT_NUM] = {
data_out_1
};
#else
ai_i8* data_outs[AI_ROAD32_OUT_NUM] = {
NULL
};
#endif

/* Activations buffers -------------------------------------------------------*/

AI_ALIGNED(32)
static uint8_t pool0[AI_ROAD32_DATA_ACTIVATION_1_SIZE];

ai_handle data_activations0[] = {pool0};

/* AI objects ----------------------------------------------------------------*/

static ai_handle road32 = AI_HANDLE_NULL;

static ai_buffer* ai_input;
static ai_buffer* ai_output;

static void ai_log_err(const ai_error err, const char *fct)
{
  /* USER CODE BEGIN log */
  // if (fct)
  //   printf("TEMPLATE - Error (%s) - type=0x%02x code=0x%02x\r\n", fct,
  //       err.type, err.code);
  // else
  //   printf("TEMPLATE - Error - type=0x%02x code=0x%02x\r\n", err.type, err.code);

  do {} while (1);
  /* USER CODE END log */
}

static int ai_boostrap(ai_handle *act_addr)
{
  ai_error err;

  /* Create and initialize an instance of the model */
  err = ai_road32_create_and_init(&road32, act_addr, NULL);
  if (err.type != AI_ERROR_NONE) {
    ai_log_err(err, "ai_road32_create_and_init");
    return -1;
  }

  ai_input = ai_road32_inputs_get(road32, NULL);
  ai_output = ai_road32_outputs_get(road32, NULL);

#if defined(AI_ROAD32_INPUTS_IN_ACTIVATIONS)
  /*  In the case where "--allocate-inputs" option is used, memory buffer can be
   *  used from the activations buffer. This is not mandatory.
   */
  for (int idx=0; idx < AI_ROAD32_IN_NUM; idx++) {
	data_ins[idx] = ai_input[idx].data;
  }
#else
  for (int idx=0; idx < AI_ROAD32_IN_NUM; idx++) {
	  ai_input[idx].data = data_ins[idx];
  }
#endif

#if defined(AI_ROAD32_OUTPUTS_IN_ACTIVATIONS)
  /*  In the case where "--allocate-outputs" option is used, memory buffer can be
   *  used from the activations buffer. This is no mandatory.
   */
  for (int idx=0; idx < AI_ROAD32_OUT_NUM; idx++) {
	data_outs[idx] = ai_output[idx].data;
  }
#else
  for (int idx=0; idx < AI_ROAD32_OUT_NUM; idx++) {
	ai_output[idx].data = data_outs[idx];
  }
#endif

  return 0;
}

static int ai_run(void)
{
  ai_i32 batch;

  batch = ai_road32_run(road32, ai_input, ai_output);
  if (batch != 1) {
    ai_log_err(ai_road32_get_error(road32),
        "ai_road32_run");
    return -1;
  }
  return 0;
}

/* USER CODE BEGIN 2 */

int acquire_and_process_data(ai_i8* data[])
{
	  char msg[50];
	  sprintf(msg, "Printing data in acquire_and_process:\r\n");
	  HAL_UART_Transmit(&huart2,(uint8_t*)msg, strlen(msg),  HAL_MAX_DELAY);

//	  ai_i8 oneSample[AI_ROAD32_IN_1_SIZE_BYTES];

	  memcpy(data[0], samples[current_sample_idx],(size_t) AI_ROAD32_IN_1_SIZE_BYTES);

	  sprintf(msg, "printing samp# %d:\r\n", current_sample_idx);
	  HAL_UART_Transmit(&huart2,(uint8_t*)msg,strlen(msg),HAL_MAX_DELAY);
	  for (int i = 0; i < 20; i++){

		  int len = sprintf(msg,"%d ", data[0][i]);

		  HAL_UART_Transmit(&huart2,(uint8_t*)msg,len,HAL_MAX_DELAY);
	  }
	  HAL_UART_Transmit(&huart2,(uint8_t*)"\r\n",2,HAL_MAX_DELAY);
	  current_sample_idx = (current_sample_idx + 1) % NUM_SAMPLES;
//    memcpy(data[0], samples[current_sample_idx], AI_ROAD32_IN_1_SIZE_BYTES);
    return 0;
}

int post_process(ai_i8* data[])
{
	int8_t q = data[0][0];                 // raw int8 output
	const float scale = 0.00390625f;
	const int zp = -128;
	float prob = (q - zp) * scale;         // dequantize to real value 0.0 → 1.0-ish

    char msg[32];
    int len = sprintf(msg, "RAW:%d  PROB:%.3f\r\n", q, prob);
//    HAL_UART_Transmit(&huart2, (uint8_t*)msg, len, HAL_MAX_DELAY);
//
//    if(prob > 0.5f) {
//           HAL_UART_Transmit(&huart2, (uint8_t*)"Normal\r\n", 8, HAL_MAX_DELAY);
//    } else {
//           HAL_UART_Transmit(&huart2, (uint8_t*)"ATTACK DETECTED\r\n", 18, HAL_MAX_DELAY);
//    }
    return 0;
}


/* USER CODE END 2 */

/* Entry points --------------------------------------------------------------*/

void MX_X_CUBE_AI_Init(void)
{
    /* USER CODE BEGIN 5 */

  ai_boostrap(data_activations0);
    /* USER CODE END 5 */
}

void MX_X_CUBE_AI_Process(void)
{
    /* USER CODE BEGIN 6 */

	  int res = -1;
	  // printf("TEMPLATE - run - main loop\r\n");
	  if (road32) {
		do {
		  res = acquire_and_process_data(data_ins);
		  if (res == 0){
			res = ai_run();
		  }
		  if (res == 0)
			res = post_process(data_outs);
		} while (res==0);
	  }
	  if (res) {
		ai_error err = {AI_ERROR_INVALID_STATE, AI_ERROR_CODE_NETWORK};
		ai_log_err(err, "Process has FAILED");
	  }
    /* USER CODE END 6 */
}
#ifdef __cplusplus
}
#endif
