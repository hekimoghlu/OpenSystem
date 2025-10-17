/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 23, 2023.
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
#ifndef AOM_COMMON_Y4MINPUT_H_
#define AOM_COMMON_Y4MINPUT_H_

#include <stdio.h>
#include "aom/aom_image.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct y4m_input y4m_input;

/*The function used to perform chroma conversion.*/
typedef void (*y4m_convert_func)(y4m_input *_y4m, unsigned char *_dst,
                                 unsigned char *_src);

struct y4m_input {
  int pic_w;
  int pic_h;
  int fps_n;
  int fps_d;
  int par_n;
  int par_d;
  char interlace;
  int src_c_dec_h;
  int src_c_dec_v;
  int dst_c_dec_h;
  int dst_c_dec_v;
  char chroma_type[16];
  /*The size of each converted frame buffer.*/
  size_t dst_buf_sz;
  /*The amount to read directly into the converted frame buffer.*/
  size_t dst_buf_read_sz;
  /*The size of the auxilliary buffer.*/
  size_t aux_buf_sz;
  /*The amount to read into the auxilliary buffer.*/
  size_t aux_buf_read_sz;
  y4m_convert_func convert;
  unsigned char *dst_buf;
  unsigned char *aux_buf;
  enum aom_img_fmt aom_fmt;
  int bps;
  unsigned int bit_depth;
  aom_color_range_t color_range;
};

/**
 * Open the input file, treating it as Y4M. |y4m_ctx| is filled in after
 * reading it. Note that |csp| should only be set for 420 input, and the input
 * chroma is shifted if necessary. The code does not support the conversion
 * from co-located to vertical. The |skip_buffer| indicates bytes that were
 * previously read from |file|, to do input-type detection; this buffer will
 * be read before the |file| is read. It is of size |num_skip|, which *must*
 * be 8 or less.
 *
 * Returns 0 on success, -1 on failure.
 */
int y4m_input_open(y4m_input *y4m_ctx, FILE *file, char *skip_buffer,
                   int num_skip, aom_chroma_sample_position_t csp,
                   int only_420);
void y4m_input_close(y4m_input *_y4m);
int y4m_input_fetch_frame(y4m_input *_y4m, FILE *_fin, aom_image_t *img);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // AOM_COMMON_Y4MINPUT_H_
