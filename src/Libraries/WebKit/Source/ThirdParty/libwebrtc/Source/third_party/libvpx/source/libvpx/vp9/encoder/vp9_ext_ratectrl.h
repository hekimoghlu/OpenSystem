/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 16, 2022.
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
#ifndef VPX_VP9_ENCODER_VP9_EXT_RATECTRL_H_
#define VPX_VP9_ENCODER_VP9_EXT_RATECTRL_H_

#include "vpx/vpx_ext_ratectrl.h"
#include "vpx/vpx_tpl.h"
#include "vp9/encoder/vp9_firstpass.h"

typedef struct EXT_RATECTRL {
  int ready;
  int ext_rdmult;
  vpx_rc_model_t model;
  vpx_rc_funcs_t funcs;
  vpx_rc_config_t ratectrl_config;
  vpx_rc_firstpass_stats_t rc_firstpass_stats;
  FILE *log_file;
} EXT_RATECTRL;

vpx_codec_err_t vp9_extrc_init(EXT_RATECTRL *ext_ratectrl);

vpx_codec_err_t vp9_extrc_create(vpx_rc_funcs_t funcs,
                                 vpx_rc_config_t ratectrl_config,
                                 EXT_RATECTRL *ext_ratectrl);

vpx_codec_err_t vp9_extrc_delete(EXT_RATECTRL *ext_ratectrl);

vpx_codec_err_t vp9_extrc_send_firstpass_stats(
    EXT_RATECTRL *ext_ratectrl, const FIRST_PASS_INFO *first_pass_info);

vpx_codec_err_t vp9_extrc_send_tpl_stats(EXT_RATECTRL *ext_ratectrl,
                                         const VpxTplGopStats *tpl_gop_stats);

vpx_codec_err_t vp9_extrc_get_encodeframe_decision(
    EXT_RATECTRL *ext_ratectrl, int gop_index,
    vpx_rc_encodeframe_decision_t *encode_frame_decision);

vpx_codec_err_t vp9_extrc_update_encodeframe_result(EXT_RATECTRL *ext_ratectrl,
                                                    int64_t bit_count,
                                                    int actual_encoding_qindex);

vpx_codec_err_t vp9_extrc_get_key_frame_decision(
    EXT_RATECTRL *ext_ratectrl,
    vpx_rc_key_frame_decision_t *key_frame_decision);

vpx_codec_err_t vp9_extrc_get_gop_decision(EXT_RATECTRL *ext_ratectrl,
                                           vpx_rc_gop_decision_t *gop_decision);

vpx_codec_err_t vp9_extrc_get_frame_rdmult(
    EXT_RATECTRL *ext_ratectrl, int show_index, int coding_index, int gop_index,
    FRAME_UPDATE_TYPE update_type, int gop_size, int use_alt_ref,
    RefCntBuffer *ref_frame_bufs[MAX_INTER_REF_FRAMES], int ref_frame_flags,
    int *rdmult);

#endif  // VPX_VP9_ENCODER_VP9_EXT_RATECTRL_H_
