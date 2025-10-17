/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 30, 2022.
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
#ifndef AM437X_VPFE_USER_H
#define AM437X_VPFE_USER_H
#include <linux/videodev2.h>
enum vpfe_ccdc_data_size {
  VPFE_CCDC_DATA_16BITS = 0,
  VPFE_CCDC_DATA_15BITS,
  VPFE_CCDC_DATA_14BITS,
  VPFE_CCDC_DATA_13BITS,
  VPFE_CCDC_DATA_12BITS,
  VPFE_CCDC_DATA_11BITS,
  VPFE_CCDC_DATA_10BITS,
  VPFE_CCDC_DATA_8BITS,
};
enum vpfe_ccdc_sample_length {
  VPFE_CCDC_SAMPLE_1PIXELS = 0,
  VPFE_CCDC_SAMPLE_2PIXELS,
  VPFE_CCDC_SAMPLE_4PIXELS,
  VPFE_CCDC_SAMPLE_8PIXELS,
  VPFE_CCDC_SAMPLE_16PIXELS,
};
enum vpfe_ccdc_sample_line {
  VPFE_CCDC_SAMPLE_1LINES = 0,
  VPFE_CCDC_SAMPLE_2LINES,
  VPFE_CCDC_SAMPLE_4LINES,
  VPFE_CCDC_SAMPLE_8LINES,
  VPFE_CCDC_SAMPLE_16LINES,
};
enum vpfe_ccdc_gamma_width {
  VPFE_CCDC_GAMMA_BITS_15_6 = 0,
  VPFE_CCDC_GAMMA_BITS_14_5,
  VPFE_CCDC_GAMMA_BITS_13_4,
  VPFE_CCDC_GAMMA_BITS_12_3,
  VPFE_CCDC_GAMMA_BITS_11_2,
  VPFE_CCDC_GAMMA_BITS_10_1,
  VPFE_CCDC_GAMMA_BITS_09_0,
};
struct vpfe_ccdc_a_law {
  unsigned char enable;
  enum vpfe_ccdc_gamma_width gamma_wd;
};
struct vpfe_ccdc_black_clamp {
  unsigned char enable;
  enum vpfe_ccdc_sample_length sample_pixel;
  enum vpfe_ccdc_sample_line sample_ln;
  unsigned short start_pixel;
  unsigned short sgain;
  unsigned short dc_sub;
};
struct vpfe_ccdc_black_compensation {
  char r;
  char gr;
  char b;
  char gb;
};
struct vpfe_ccdc_config_params_raw {
  enum vpfe_ccdc_data_size data_sz;
  struct vpfe_ccdc_a_law alaw;
  struct vpfe_ccdc_black_clamp blk_clamp;
  struct vpfe_ccdc_black_compensation blk_comp;
};
#define VIDIOC_AM437X_CCDC_CFG _IOW('V', BASE_VIDIOC_PRIVATE + 1, void *)
#endif
