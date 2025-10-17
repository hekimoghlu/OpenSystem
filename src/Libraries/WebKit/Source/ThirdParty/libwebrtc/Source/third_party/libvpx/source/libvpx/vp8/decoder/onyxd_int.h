/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 1, 2022.
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
#ifndef VPX_VP8_DECODER_ONYXD_INT_H_
#define VPX_VP8_DECODER_ONYXD_INT_H_

#include <assert.h>

#include "vpx_config.h"
#include "vpx_util/vpx_pthread.h"
#include "vp8/common/onyxd.h"
#include "treereader.h"
#include "vp8/common/onyxc_int.h"
#include "vp8/common/threading.h"

#if CONFIG_ERROR_CONCEALMENT
#include "ec_types.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  int ithread;
  void *ptr1;
  void *ptr2;
} DECODETHREAD_DATA;

typedef struct {
  MACROBLOCKD mbd;
} MB_ROW_DEC;

typedef struct {
  int enabled;
  unsigned int count;
  const unsigned char *ptrs[MAX_PARTITIONS];
  unsigned int sizes[MAX_PARTITIONS];
} FRAGMENT_DATA;

#define MAX_FB_MT_DEC 32

struct frame_buffers {
  /*
   * this struct will be populated with frame buffer management
   * info in future commits. */

  /* decoder instances */
  struct VP8D_COMP *pbi[MAX_FB_MT_DEC];
};

typedef struct VP8D_COMP {
  DECLARE_ALIGNED(16, MACROBLOCKD, mb);

  YV12_BUFFER_CONFIG *dec_fb_ref[NUM_YV12_BUFFERS];

  DECLARE_ALIGNED(16, VP8_COMMON, common);

  /* the last partition will be used for the modes/mvs */
  vp8_reader mbc[MAX_PARTITIONS];

  VP8D_CONFIG oxcf;

  FRAGMENT_DATA fragments;

#if CONFIG_MULTITHREAD
  /* variable for threading */

  vpx_atomic_int b_multithreaded_rd;
  int max_threads;
  int current_mb_col_main;
  unsigned int decoding_thread_count;
  int allocated_decoding_thread_count;

  int mt_baseline_filter_level[MAX_MB_SEGMENTS];
  int sync_range;
  /* Each row remembers its already decoded column. */
  vpx_atomic_int *mt_current_mb_col;

  unsigned char **mt_yabove_row; /* mb_rows x width */
  unsigned char **mt_uabove_row;
  unsigned char **mt_vabove_row;
  unsigned char **mt_yleft_col; /* mb_rows x 16 */
  unsigned char **mt_uleft_col; /* mb_rows x 8 */
  unsigned char **mt_vleft_col; /* mb_rows x 8 */

  MB_ROW_DEC *mb_row_di;
  DECODETHREAD_DATA *de_thread_data;

  pthread_t *h_decoding_thread;
  vp8_sem_t *h_event_start_decoding;
  vp8_sem_t h_event_end_decoding;
/* end of threading data */
#endif

  int ready_for_new_data;

  vp8_prob prob_intra;
  vp8_prob prob_last;
  vp8_prob prob_gf;
  vp8_prob prob_skip_false;

#if CONFIG_ERROR_CONCEALMENT
  MB_OVERLAP *overlaps;
  /* the mb num from which modes and mvs (first partition) are corrupt */
  unsigned int mvs_corrupt_from_mb;
#endif
  int ec_enabled;
  int ec_active;
  int decoded_key_frame;
  int independent_partitions;
  int frame_corrupt_residual;

  vpx_decrypt_cb decrypt_cb;
  void *decrypt_state;
#if CONFIG_MULTITHREAD
  // Restart threads on next frame if set to 1.
  // This is set when error happens in multithreaded decoding and all threads
  // are shut down.
  int restart_threads;
#endif
} VP8D_COMP;

void vp8cx_init_de_quantizer(VP8D_COMP *pbi);
void vp8_mb_init_dequantizer(VP8D_COMP *pbi, MACROBLOCKD *xd);
int vp8_decode_frame(VP8D_COMP *pbi);

int vp8_create_decoder_instances(struct frame_buffers *fb, VP8D_CONFIG *oxcf);
int vp8_remove_decoder_instances(struct frame_buffers *fb);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // VPX_VP8_DECODER_ONYXD_INT_H_
