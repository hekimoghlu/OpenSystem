/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 26, 2023.
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
#ifndef CALL_ADAPTATION_TEST_FAKE_FRAME_RATE_PROVIDER_H_
#define CALL_ADAPTATION_TEST_FAKE_FRAME_RATE_PROVIDER_H_

#include <string>
#include <vector>

#include "test/gmock.h"
#include "video/video_stream_encoder_observer.h"

namespace webrtc {

class MockVideoStreamEncoderObserver : public VideoStreamEncoderObserver {
 public:
  MOCK_METHOD(void, OnEncodedFrameTimeMeasured, (int, int), (override));
  MOCK_METHOD(void, OnIncomingFrame, (int, int), (override));
  MOCK_METHOD(void,
              OnSendEncodedImage,
              (const EncodedImage&, const CodecSpecificInfo*),
              (override));
  MOCK_METHOD(void,
              OnEncoderImplementationChanged,
              (EncoderImplementation),
              (override));
  MOCK_METHOD(void, OnFrameDropped, (DropReason), (override));
  MOCK_METHOD(void,
              OnEncoderReconfigured,
              (const VideoEncoderConfig&, const std::vector<VideoStream>&),
              (override));
  MOCK_METHOD(void,
              OnAdaptationChanged,
              (VideoAdaptationReason,
               const VideoAdaptationCounters&,
               const VideoAdaptationCounters&),
              (override));
  MOCK_METHOD(void, ClearAdaptationStats, (), (override));
  MOCK_METHOD(void,
              UpdateAdaptationSettings,
              (AdaptationSettings, AdaptationSettings),
              (override));
  MOCK_METHOD(void, OnMinPixelLimitReached, (), (override));
  MOCK_METHOD(void, OnInitialQualityResolutionAdaptDown, (), (override));
  MOCK_METHOD(void, OnSuspendChange, (bool), (override));
  MOCK_METHOD(void,
              OnBitrateAllocationUpdated,
              (const VideoCodec&, const VideoBitrateAllocation&),
              (override));
  MOCK_METHOD(void, OnEncoderInternalScalerUpdate, (bool), (override));
  MOCK_METHOD(int, GetInputFrameRate, (), (const, override));
};

class FakeFrameRateProvider : public MockVideoStreamEncoderObserver {
 public:
  FakeFrameRateProvider();
  void set_fps(int fps);
};

}  // namespace webrtc

#endif  // CALL_ADAPTATION_TEST_FAKE_FRAME_RATE_PROVIDER_H_
