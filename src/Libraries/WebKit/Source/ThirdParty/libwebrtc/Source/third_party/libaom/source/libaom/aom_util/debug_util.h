/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 15, 2021.
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
#ifndef AOM_AOM_UTIL_DEBUG_UTIL_H_
#define AOM_AOM_UTIL_DEBUG_UTIL_H_

#include "config/aom_config.h"

#include "aom_dsp/prob.h"

#ifdef __cplusplus
extern "C" {
#endif

void aom_bitstream_queue_set_frame_write(int frame_idx);
int aom_bitstream_queue_get_frame_write(void);
void aom_bitstream_queue_set_frame_read(int frame_idx);
int aom_bitstream_queue_get_frame_read(void);

#if CONFIG_BITSTREAM_DEBUG
/* This is a debug tool used to detect bitstream error. On encoder side, it
 * pushes each bit and probability into a queue before the bit is written into
 * the Arithmetic coder. On decoder side, whenever a bit is read out from the
 * Arithmetic coder, it pops out the reference bit and probability from the
 * queue as well. If the two results do not match, this debug tool will report
 * an error.  This tool can be used to pin down the bitstream error precisely.
 * By combining gdb's backtrace method, we can detect which module causes the
 * bitstream error. */
int bitstream_queue_get_write(void);
int bitstream_queue_get_read(void);
void bitstream_queue_record_write(void);
void bitstream_queue_reset_write(void);
void bitstream_queue_pop(int *result, aom_cdf_prob *cdf, int *nsymbs);
void bitstream_queue_push(int result, const aom_cdf_prob *cdf, int nsymbs);
void bitstream_queue_set_skip_write(int skip);
void bitstream_queue_set_skip_read(int skip);
#endif  // CONFIG_BITSTREAM_DEBUG

#if CONFIG_MISMATCH_DEBUG
void mismatch_move_frame_idx_w();
void mismatch_move_frame_idx_r();
void mismatch_reset_frame(int num_planes);
void mismatch_record_block_pre(const uint8_t *src, int src_stride,
                               int frame_offset, int plane, int pixel_c,
                               int pixel_r, int blk_w, int blk_h, int highbd);
void mismatch_record_block_tx(const uint8_t *src, int src_stride,
                              int frame_offset, int plane, int pixel_c,
                              int pixel_r, int blk_w, int blk_h, int highbd);
void mismatch_check_block_pre(const uint8_t *src, int src_stride,
                              int frame_offset, int plane, int pixel_c,
                              int pixel_r, int blk_w, int blk_h, int highbd);
void mismatch_check_block_tx(const uint8_t *src, int src_stride,
                             int frame_offset, int plane, int pixel_c,
                             int pixel_r, int blk_w, int blk_h, int highbd);
#endif  // CONFIG_MISMATCH_DEBUG

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // AOM_AOM_UTIL_DEBUG_UTIL_H_
