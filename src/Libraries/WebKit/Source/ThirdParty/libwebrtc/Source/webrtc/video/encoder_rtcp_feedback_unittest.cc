/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 4, 2024.
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
#include "video/encoder_rtcp_feedback.h"

#include <memory>

#include "api/environment/environment_factory.h"
#include "test/gmock.h"
#include "test/gtest.h"
#include "video/test/mock_video_stream_encoder.h"

using ::testing::_;
using ::testing::ElementsAre;

namespace webrtc {

class VideoEncoderFeedbackKeyframeTestBase : public ::testing::Test {
 public:
  VideoEncoderFeedbackKeyframeTestBase(bool per_layer_pli_handling,
                                       std::vector<uint32_t> ssrcs)
      : simulated_clock_(123456789),
        encoder_(),
        encoder_rtcp_feedback_(CreateEnvironment(&simulated_clock_),
                               per_layer_pli_handling,
                               ssrcs,
                               &encoder_,
                               nullptr) {}

 protected:
  static const uint32_t kSsrc = 1234;
  static const uint32_t kOtherSsrc = 4321;

  SimulatedClock simulated_clock_;
  ::testing::StrictMock<MockVideoStreamEncoder> encoder_;
  EncoderRtcpFeedback encoder_rtcp_feedback_;
};

class VideoEncoderFeedbackKeyframeTest
    : public VideoEncoderFeedbackKeyframeTestBase {
 public:
  VideoEncoderFeedbackKeyframeTest()
      : VideoEncoderFeedbackKeyframeTestBase(
            /*per_layer_pli_handling=*/false,
            {VideoEncoderFeedbackKeyframeTestBase::kSsrc}) {}
};

TEST_F(VideoEncoderFeedbackKeyframeTest, CreateAndTriggerRequests) {
  EXPECT_CALL(encoder_, SendKeyFrame(_)).Times(1);
  encoder_rtcp_feedback_.OnReceivedIntraFrameRequest(kSsrc);
}

TEST_F(VideoEncoderFeedbackKeyframeTest, TooManyOnReceivedIntraFrameRequest) {
  EXPECT_CALL(encoder_, SendKeyFrame(_)).Times(1);
  encoder_rtcp_feedback_.OnReceivedIntraFrameRequest(kSsrc);
  encoder_rtcp_feedback_.OnReceivedIntraFrameRequest(kSsrc);
  simulated_clock_.AdvanceTimeMilliseconds(10);
  encoder_rtcp_feedback_.OnReceivedIntraFrameRequest(kSsrc);

  EXPECT_CALL(encoder_, SendKeyFrame(_)).Times(1);
  simulated_clock_.AdvanceTimeMilliseconds(300);
  encoder_rtcp_feedback_.OnReceivedIntraFrameRequest(kSsrc);
  encoder_rtcp_feedback_.OnReceivedIntraFrameRequest(kSsrc);
  encoder_rtcp_feedback_.OnReceivedIntraFrameRequest(kSsrc);
}

class VideoEncoderFeedbackKeyframePerLayerPliTest
    : public VideoEncoderFeedbackKeyframeTestBase {
 public:
  VideoEncoderFeedbackKeyframePerLayerPliTest()
      : VideoEncoderFeedbackKeyframeTestBase(
            /*per_layer_pli_handling=*/true,
            {VideoEncoderFeedbackKeyframeTestBase::kSsrc,
             VideoEncoderFeedbackKeyframeTestBase::kOtherSsrc}) {}
};

TEST_F(VideoEncoderFeedbackKeyframePerLayerPliTest, CreateAndTriggerRequests) {
  EXPECT_CALL(encoder_,
              SendKeyFrame(ElementsAre(VideoFrameType::kVideoFrameKey,
                                       VideoFrameType::kVideoFrameDelta)))
      .Times(1);
  EXPECT_CALL(encoder_,
              SendKeyFrame(ElementsAre(VideoFrameType::kVideoFrameDelta,
                                       VideoFrameType::kVideoFrameKey)))
      .Times(1);
  encoder_rtcp_feedback_.OnReceivedIntraFrameRequest(kSsrc);
  encoder_rtcp_feedback_.OnReceivedIntraFrameRequest(kOtherSsrc);
}

TEST_F(VideoEncoderFeedbackKeyframePerLayerPliTest,
       TooManyOnReceivedIntraFrameRequest) {
  EXPECT_CALL(encoder_,
              SendKeyFrame(ElementsAre(VideoFrameType::kVideoFrameKey,
                                       VideoFrameType::kVideoFrameDelta)))
      .Times(1);
  EXPECT_CALL(encoder_,
              SendKeyFrame(ElementsAre(VideoFrameType::kVideoFrameDelta,
                                       VideoFrameType::kVideoFrameKey)))
      .Times(1);
  encoder_rtcp_feedback_.OnReceivedIntraFrameRequest(kSsrc);
  encoder_rtcp_feedback_.OnReceivedIntraFrameRequest(kSsrc);
  encoder_rtcp_feedback_.OnReceivedIntraFrameRequest(kOtherSsrc);
  simulated_clock_.AdvanceTimeMilliseconds(10);
  encoder_rtcp_feedback_.OnReceivedIntraFrameRequest(kSsrc);
  encoder_rtcp_feedback_.OnReceivedIntraFrameRequest(kOtherSsrc);

  EXPECT_CALL(encoder_,
              SendKeyFrame(ElementsAre(VideoFrameType::kVideoFrameKey,
                                       VideoFrameType::kVideoFrameDelta)))
      .Times(1);
  EXPECT_CALL(encoder_,
              SendKeyFrame(ElementsAre(VideoFrameType::kVideoFrameDelta,
                                       VideoFrameType::kVideoFrameKey)))
      .Times(1);
  simulated_clock_.AdvanceTimeMilliseconds(300);
  encoder_rtcp_feedback_.OnReceivedIntraFrameRequest(kSsrc);
  encoder_rtcp_feedback_.OnReceivedIntraFrameRequest(kSsrc);
  encoder_rtcp_feedback_.OnReceivedIntraFrameRequest(kSsrc);
  encoder_rtcp_feedback_.OnReceivedIntraFrameRequest(kOtherSsrc);
  encoder_rtcp_feedback_.OnReceivedIntraFrameRequest(kOtherSsrc);
  encoder_rtcp_feedback_.OnReceivedIntraFrameRequest(kOtherSsrc);
}

}  // namespace webrtc
