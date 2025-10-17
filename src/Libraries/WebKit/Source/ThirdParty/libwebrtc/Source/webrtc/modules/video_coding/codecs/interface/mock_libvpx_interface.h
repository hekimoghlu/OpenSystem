/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 7, 2024.
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
#ifndef MODULES_VIDEO_CODING_CODECS_INTERFACE_MOCK_LIBVPX_INTERFACE_H_
#define MODULES_VIDEO_CODING_CODECS_INTERFACE_MOCK_LIBVPX_INTERFACE_H_

#include "modules/video_coding/codecs/interface/libvpx_interface.h"
#include "test/gmock.h"
#include "test/gtest.h"

namespace webrtc {

class MockLibvpxInterface : public LibvpxInterface {
 public:
  MOCK_METHOD(
      vpx_image_t*,
      img_alloc,
      (vpx_image_t*, vpx_img_fmt_t, unsigned int, unsigned int, unsigned int),
      (const, override));
  MOCK_METHOD(vpx_image_t*,
              img_wrap,
              (vpx_image_t*,
               vpx_img_fmt_t,
               unsigned int,
               unsigned int,
               unsigned int,
               unsigned char*),
              (const, override));
  MOCK_METHOD(void, img_free, (vpx_image_t * img), (const, override));
  MOCK_METHOD(vpx_codec_err_t,
              codec_enc_config_set,
              (vpx_codec_ctx_t*, const vpx_codec_enc_cfg_t*),
              (const, override));
  MOCK_METHOD(vpx_codec_err_t,
              codec_enc_config_default,
              (vpx_codec_iface_t*, vpx_codec_enc_cfg_t*, unsigned int),
              (const, override));
  MOCK_METHOD(vpx_codec_err_t,
              codec_enc_init,
              (vpx_codec_ctx_t*,
               vpx_codec_iface_t*,
               const vpx_codec_enc_cfg_t*,
               vpx_codec_flags_t),
              (const, override));
  MOCK_METHOD(vpx_codec_err_t,
              codec_enc_init_multi,
              (vpx_codec_ctx_t*,
               vpx_codec_iface_t*,
               vpx_codec_enc_cfg_t*,
               int,
               vpx_codec_flags_t,
               vpx_rational_t*),
              (const, override));
  MOCK_METHOD(vpx_codec_err_t,
              codec_destroy,
              (vpx_codec_ctx_t*),
              (const, override));
  MOCK_METHOD(vpx_codec_err_t,
              codec_control,
              (vpx_codec_ctx_t*, vp8e_enc_control_id, uint32_t),
              (const, override));
  MOCK_METHOD(vpx_codec_err_t,
              codec_control,
              (vpx_codec_ctx_t*, vp8e_enc_control_id, int),
              (const, override));
  MOCK_METHOD(vpx_codec_err_t,
              codec_control,
              (vpx_codec_ctx_t*, vp8e_enc_control_id, int*),
              (const, override));
  MOCK_METHOD(vpx_codec_err_t,
              codec_control,
              (vpx_codec_ctx_t*, vp8e_enc_control_id, vpx_roi_map*),
              (const, override));
  MOCK_METHOD(vpx_codec_err_t,
              codec_control,
              (vpx_codec_ctx_t*, vp8e_enc_control_id, vpx_active_map*),
              (const, override));
  MOCK_METHOD(vpx_codec_err_t,
              codec_control,
              (vpx_codec_ctx_t*, vp8e_enc_control_id, vpx_scaling_mode*),
              (const, override));
  MOCK_METHOD(vpx_codec_err_t,
              codec_control,
              (vpx_codec_ctx_t*, vp8e_enc_control_id, vpx_svc_extra_cfg_t*),
              (const, override));
  MOCK_METHOD(vpx_codec_err_t,
              codec_control,
              (vpx_codec_ctx_t*, vp8e_enc_control_id, vpx_svc_frame_drop_t*),
              (const, override));
  MOCK_METHOD(vpx_codec_err_t,
              codec_control,
              (vpx_codec_ctx_t*, vp8e_enc_control_id, void*),
              (const, override));
  MOCK_METHOD(vpx_codec_err_t,
              codec_control,
              (vpx_codec_ctx_t*, vp8e_enc_control_id, vpx_svc_layer_id_t*),
              (const, override));
  MOCK_METHOD(vpx_codec_err_t,
              codec_control,
              (vpx_codec_ctx_t*,
               vp8e_enc_control_id,
               vpx_svc_ref_frame_config_t*),
              (const, override));
  MOCK_METHOD(vpx_codec_err_t,
              codec_control,
              (vpx_codec_ctx_t*,
               vp8e_enc_control_id,
               vpx_svc_spatial_layer_sync_t*),
              (const, override));
  MOCK_METHOD(vpx_codec_err_t,
              codec_control,
              (vpx_codec_ctx_t*, vp8e_enc_control_id, vpx_rc_funcs_t*),
              (const, override));
  MOCK_METHOD(vpx_codec_err_t,
              codec_encode,
              (vpx_codec_ctx_t*,
               const vpx_image_t*,
               vpx_codec_pts_t,
               uint64_t,
               vpx_enc_frame_flags_t,
               uint64_t),
              (const, override));
  MOCK_METHOD(const vpx_codec_cx_pkt_t*,
              codec_get_cx_data,
              (vpx_codec_ctx_t*, vpx_codec_iter_t*),
              (const, override));
  MOCK_METHOD(const char*,
              codec_error_detail,
              (vpx_codec_ctx_t*),
              (const, override));
  MOCK_METHOD(const char*, codec_error, (vpx_codec_ctx_t*), (const, override));
  MOCK_METHOD(const char*,
              codec_err_to_string,
              (vpx_codec_err_t),
              (const, override));
};

}  // namespace webrtc

#endif  // MODULES_VIDEO_CODING_CODECS_INTERFACE_MOCK_LIBVPX_INTERFACE_H_
