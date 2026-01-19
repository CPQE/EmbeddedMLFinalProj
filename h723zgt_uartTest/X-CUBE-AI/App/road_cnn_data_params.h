/**
  ******************************************************************************
  * @file    road_cnn_data_params.h
  * @author  AST Embedded Analytics Research Platform
  * @date    2026-01-17T11:05:09-0800
  * @brief   AI Tool Automatic Code Generator for Embedded NN computing
  ******************************************************************************
  * Copyright (c) 2026 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  ******************************************************************************
  */

#ifndef ROAD_CNN_DATA_PARAMS_H
#define ROAD_CNN_DATA_PARAMS_H

#include "ai_platform.h"

/*
#define AI_ROAD_CNN_DATA_WEIGHTS_PARAMS \
  (AI_HANDLE_PTR(&ai_road_cnn_data_weights_params[1]))
*/

#define AI_ROAD_CNN_DATA_CONFIG               (NULL)


#define AI_ROAD_CNN_DATA_ACTIVATIONS_SIZES \
  { 131984, }
#define AI_ROAD_CNN_DATA_ACTIVATIONS_SIZE     (131984)
#define AI_ROAD_CNN_DATA_ACTIVATIONS_COUNT    (1)
#define AI_ROAD_CNN_DATA_ACTIVATION_1_SIZE    (131984)



#define AI_ROAD_CNN_DATA_WEIGHTS_SIZES \
  { 12036, }
#define AI_ROAD_CNN_DATA_WEIGHTS_SIZE         (12036)
#define AI_ROAD_CNN_DATA_WEIGHTS_COUNT        (1)
#define AI_ROAD_CNN_DATA_WEIGHT_1_SIZE        (12036)



#define AI_ROAD_CNN_DATA_ACTIVATIONS_TABLE_GET() \
  (&g_road_cnn_activations_table[1])

extern ai_handle g_road_cnn_activations_table[1 + 2];



#define AI_ROAD_CNN_DATA_WEIGHTS_TABLE_GET() \
  (&g_road_cnn_weights_table[1])

extern ai_handle g_road_cnn_weights_table[1 + 2];


#endif    /* ROAD_CNN_DATA_PARAMS_H */
