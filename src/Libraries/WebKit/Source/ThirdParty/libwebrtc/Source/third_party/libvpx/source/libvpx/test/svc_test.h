/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 15, 2024.
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
#ifndef VPX_TEST_SVC_TEST_H_
#define VPX_TEST_SVC_TEST_H_

#include "./vpx_config.h"
#include "gtest/gtest.h"
#include "test/codec_factory.h"
#include "test/encode_test_driver.h"
#include "test/i420_video_source.h"
#include "test/util.h"
#include "test/y4m_video_source.h"
#include "vpx/vpx_codec.h"
#include "vpx_ports/bitops.h"

namespace svc_test {
class OnePassCbrSvc : public ::libvpx_test::EncoderTest {
 public:
  explicit OnePassCbrSvc(const ::libvpx_test::CodecFactory *codec)
      : EncoderTest(codec), base_speed_setting_(0), speed_setting_(0),
        superframe_count_(0), temporal_layer_id_(0), number_temporal_layers_(0),
        number_spatial_layers_(0) {
    memset(&svc_params_, 0, sizeof(svc_params_));
    memset(bits_in_buffer_model_, 0,
           sizeof(bits_in_buffer_model_[0]) * VPX_MAX_LAYERS);
    memset(layer_target_avg_bandwidth_, 0,
           sizeof(layer_target_avg_bandwidth_[0]) * VPX_MAX_LAYERS);
  }

 protected:
  ~OnePassCbrSvc() override {}

  virtual void SetConfig(const int num_temporal_layer) = 0;

  virtual void SetSvcConfig(const int num_spatial_layer,
                            const int num_temporal_layer);

  virtual void PreEncodeFrameHookSetup(::libvpx_test::VideoSource *video,
                                       ::libvpx_test::Encoder *encoder);

  void PostEncodeFrameHook(::libvpx_test::Encoder *encoder) override;

  virtual void AssignLayerBitrates();

  void MismatchHook(const vpx_image_t *, const vpx_image_t *) override {}

  vpx_svc_extra_cfg_t svc_params_;
  int64_t bits_in_buffer_model_[VPX_MAX_LAYERS];
  int layer_target_avg_bandwidth_[VPX_MAX_LAYERS];
  int base_speed_setting_;
  int speed_setting_;
  int superframe_count_;
  int temporal_layer_id_;
  int number_temporal_layers_;
  int number_spatial_layers_;
};
}  // namespace svc_test

#endif  // VPX_TEST_SVC_TEST_H_
