/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 6, 2022.
 *
 * Licensed under the Apache License, Version 2.0 (the ""License"");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at:
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an ""AS IS"" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Please contact NeXTHub Corporation, 651 N Broad St, Suite 201, 
 * Middletown, DE 19709, New Castle County, USA.
 *
 */
#ifndef AOM_AV1_COMMON_TIMING_H_
#define AOM_AV1_COMMON_TIMING_H_

#include "aom/aom_integer.h"
#include "av1/common/enums.h"

#define MAX_NUM_OP_POINTS 32

typedef struct aom_timing {
  uint32_t num_units_in_display_tick;
  uint32_t time_scale;
  int equal_picture_interval;
  uint32_t num_ticks_per_picture;
} aom_timing_info_t;

typedef struct aom_dec_model_info {
  uint32_t num_units_in_decoding_tick;
  int encoder_decoder_buffer_delay_length;
  int buffer_removal_time_length;
  int frame_presentation_time_length;
} aom_dec_model_info_t;

typedef struct aom_dec_model_op_parameters {
  int decoder_model_param_present_flag;
  int64_t bitrate;
  int64_t buffer_size;
  uint32_t decoder_buffer_delay;
  uint32_t encoder_buffer_delay;
  int low_delay_mode_flag;
  int display_model_param_present_flag;
  int initial_display_delay;
} aom_dec_model_op_parameters_t;

void av1_set_aom_dec_model_info(aom_dec_model_info_t *decoder_model);

void av1_set_dec_model_op_parameters(aom_dec_model_op_parameters_t *op_params);

void av1_set_resource_availability_parameters(
    aom_dec_model_op_parameters_t *op_params);

int64_t av1_max_level_bitrate(BITSTREAM_PROFILE seq_profile, int seq_level_idx,
                              int seq_tier);

#endif  // AOM_AV1_COMMON_TIMING_H_
