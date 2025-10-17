/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 18, 2023.
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
#ifndef VIDEO_TEST_MOCK_VIDEO_STREAM_ENCODER_H_
#define VIDEO_TEST_MOCK_VIDEO_STREAM_ENCODER_H_

#include <vector>

#include "test/gmock.h"
#include "video/video_stream_encoder_interface.h"

namespace webrtc {

class MockVideoStreamEncoder : public VideoStreamEncoderInterface {
 public:
  MOCK_METHOD(void,
              AddAdaptationResource,
              (rtc::scoped_refptr<Resource>),
              (override));
  MOCK_METHOD(std::vector<rtc::scoped_refptr<Resource>>,
              GetAdaptationResources,
              (),
              (override));
  MOCK_METHOD(void,
              SetSource,
              (rtc::VideoSourceInterface<VideoFrame>*,
               const DegradationPreference&),
              (override));
  MOCK_METHOD(void, SetSink, (EncoderSink*, bool), (override));
  MOCK_METHOD(void, SetStartBitrate, (int), (override));
  MOCK_METHOD(void,
              SendKeyFrame,
              (const std::vector<VideoFrameType>&),
              (override));
  MOCK_METHOD(void,
              OnLossNotification,
              (const VideoEncoder::LossNotification&),
              (override));
  MOCK_METHOD(void,
              OnBitrateUpdated,
              (DataRate, DataRate, DataRate, uint8_t, int64_t, double),
              (override));
  MOCK_METHOD(void,
              SetFecControllerOverride,
              (FecControllerOverride*),
              (override));
  MOCK_METHOD(void, Stop, (), (override));

  MOCK_METHOD(void,
              MockedConfigureEncoder,
              (const VideoEncoderConfig&, size_t));
  MOCK_METHOD(void,
              MockedConfigureEncoder,
              (const VideoEncoderConfig&, size_t, SetParametersCallback));
  // gtest generates implicit copy which is not allowed on VideoEncoderConfig,
  // so we can't mock ConfigureEncoder directly.
  void ConfigureEncoder(VideoEncoderConfig config,
                        size_t max_data_payload_length) {
    MockedConfigureEncoder(config, max_data_payload_length);
  }
  void ConfigureEncoder(VideoEncoderConfig config,
                        size_t max_data_payload_length,
                        SetParametersCallback) {
    MockedConfigureEncoder(config, max_data_payload_length);
  }
};

}  // namespace webrtc

#endif  // VIDEO_TEST_MOCK_VIDEO_STREAM_ENCODER_H_
