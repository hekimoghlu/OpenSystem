/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 11, 2021.
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
#ifndef TEST_FAKE_VP8_ENCODER_H_
#define TEST_FAKE_VP8_ENCODER_H_

#include <stddef.h>
#include <stdint.h>

#include <memory>

#include "api/environment/environment.h"
#include "api/fec_controller_override.h"
#include "api/sequence_checker.h"
#include "api/video/encoded_image.h"
#include "api/video_codecs/video_codec.h"
#include "api/video_codecs/video_encoder.h"
#include "api/video_codecs/vp8_frame_buffer_controller.h"
#include "api/video_codecs/vp8_temporal_layers.h"
#include "modules/video_coding/include/video_codec_interface.h"
#include "rtc_base/thread_annotations.h"
#include "system_wrappers/include/clock.h"
#include "test/fake_encoder.h"

namespace webrtc {
namespace test {

class FakeVp8Encoder : public FakeEncoder {
 public:
  explicit FakeVp8Encoder(const Environment& env);
  [[deprecated]] explicit FakeVp8Encoder(Clock* clock);
  virtual ~FakeVp8Encoder() = default;

  int32_t InitEncode(const VideoCodec* config,
                     const Settings& settings) override;

  int32_t Release() override;

  EncoderInfo GetEncoderInfo() const override;

 private:
  CodecSpecificInfo PopulateCodecSpecific(size_t size_bytes,
                                          VideoFrameType frame_type,
                                          int stream_idx,
                                          uint32_t timestamp);

  CodecSpecificInfo EncodeHook(
      EncodedImage& encoded_image,
      rtc::scoped_refptr<EncodedImageBuffer> buffer) override;

  SequenceChecker sequence_checker_;

  class FakeFecControllerOverride : public FecControllerOverride {
   public:
    ~FakeFecControllerOverride() override = default;

    void SetFecAllowed(bool fec_allowed) override {}
  };

  FakeFecControllerOverride fec_controller_override_
      RTC_GUARDED_BY(sequence_checker_);

  std::unique_ptr<Vp8FrameBufferController> frame_buffer_controller_
      RTC_GUARDED_BY(sequence_checker_);
};

}  // namespace test
}  // namespace webrtc

#endif  // TEST_FAKE_VP8_ENCODER_H_
