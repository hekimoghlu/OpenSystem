/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 4, 2023.
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
#ifndef API_TEST_MOCK_VIDEO_ENCODER_H_
#define API_TEST_MOCK_VIDEO_ENCODER_H_

#include <cstddef>
#include <cstdint>
#include <vector>

#include "api/fec_controller_override.h"
#include "api/video/encoded_image.h"
#include "api/video/video_frame.h"
#include "api/video/video_frame_type.h"
#include "api/video_codecs/video_codec.h"
#include "api/video_codecs/video_encoder.h"
#include "test/gmock.h"

namespace webrtc {

class MockEncodedImageCallback : public EncodedImageCallback {
 public:
  MOCK_METHOD(Result,
              OnEncodedImage,
              (const EncodedImage&, const CodecSpecificInfo*),
              (override));
  MOCK_METHOD(void, OnDroppedFrame, (DropReason reason), (override));
};

class MockVideoEncoder : public VideoEncoder {
 public:
  MOCK_METHOD(void,
              SetFecControllerOverride,
              (FecControllerOverride*),
              (override));
  MOCK_METHOD(int32_t,
              InitEncode,
              (const VideoCodec*, int32_t numberOfCores, size_t maxPayloadSize),
              (override));
  MOCK_METHOD(int32_t,
              InitEncode,
              (const VideoCodec*, const VideoEncoder::Settings& settings),
              (override));

  MOCK_METHOD(int32_t,
              Encode,
              (const VideoFrame& inputImage,
               const std::vector<VideoFrameType>*),
              (override));
  MOCK_METHOD(int32_t,
              RegisterEncodeCompleteCallback,
              (EncodedImageCallback*),
              (override));
  MOCK_METHOD(int32_t, Release, (), (override));
  MOCK_METHOD(void,
              SetRates,
              (const RateControlParameters& parameters),
              (override));
  MOCK_METHOD(void,
              OnPacketLossRateUpdate,
              (float packet_loss_rate),
              (override));
  MOCK_METHOD(void, OnRttUpdate, (int64_t rtt_ms), (override));
  MOCK_METHOD(void,
              OnLossNotification,
              (const LossNotification& loss_notification),
              (override));
  MOCK_METHOD(EncoderInfo, GetEncoderInfo, (), (const, override));
};

}  // namespace webrtc

#endif  // API_TEST_MOCK_VIDEO_ENCODER_H_
