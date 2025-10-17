/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 19, 2025.
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
#ifndef VPX_VP9_VP9_DX_IFACE_H_
#define VPX_VP9_VP9_DX_IFACE_H_

#include "vp9/decoder/vp9_decoder.h"

typedef vpx_codec_stream_info_t vp9_stream_info_t;

struct vpx_codec_alg_priv {
  vpx_codec_priv_t base;
  vpx_codec_dec_cfg_t cfg;
  vp9_stream_info_t si;
  VP9Decoder *pbi;
  void *user_priv;
  int postproc_cfg_set;
  vp8_postproc_cfg_t postproc_cfg;
  vpx_decrypt_cb decrypt_cb;
  void *decrypt_state;
  vpx_image_t img;
  int img_avail;
  int flushed;
  int invert_tile_order;
  int last_show_frame;  // Index of last output frame.
  int byte_alignment;
  int skip_loop_filter;

  int need_resync;  // wait for key/intra-only frame
  // BufferPool that holds all reference frames.
  BufferPool *buffer_pool;

  // External frame buffer info to save for VP9 common.
  void *ext_priv;  // Private data associated with the external frame buffers.
  vpx_get_frame_buffer_cb_fn_t get_ext_fb_cb;
  vpx_release_frame_buffer_cb_fn_t release_ext_fb_cb;

  // Allow for decoding up to a given spatial layer for SVC stream.
  int svc_decoding;
  int svc_spatial_layer;
  int row_mt;
  int lpf_opt;
};

#endif  // VPX_VP9_VP9_DX_IFACE_H_
