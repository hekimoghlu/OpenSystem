/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 30, 2022.
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
#include "api/frame_transformer_factory.h"

#include <cstdio>
#include <memory>
#include <utility>
#include <vector>

#include "absl/memory/memory.h"
#include "api/call/transport.h"
#include "api/test/mock_frame_transformer.h"
#include "api/test/mock_transformable_audio_frame.h"
#include "api/test/mock_transformable_video_frame.h"
#include "call/video_receive_stream.h"
#include "modules/rtp_rtcp/source/rtp_descriptor_authentication.h"
#include "rtc_base/event.h"
#include "test/gmock.h"
#include "test/gtest.h"

namespace webrtc {
namespace {

using testing::Each;
using testing::ElementsAreArray;
using testing::NiceMock;
using testing::Return;
using testing::ReturnRef;

TEST(FrameTransformerFactory, CloneAudioFrame) {
  NiceMock<MockTransformableAudioFrame> original_frame;
  uint8_t data[10];
  std::fill_n(data, 10, 5);
  rtc::ArrayView<uint8_t> data_view(data);
  ON_CALL(original_frame, GetData()).WillByDefault(Return(data_view));
  auto cloned_frame = CloneAudioFrame(&original_frame);

  EXPECT_THAT(cloned_frame->GetData(), ElementsAreArray(data));
}

TEST(FrameTransformerFactory, CloneVideoFrame) {
  NiceMock<MockTransformableVideoFrame> original_frame;
  uint8_t data[10];
  std::fill_n(data, 10, 5);
  rtc::ArrayView<uint8_t> data_view(data);
  EXPECT_CALL(original_frame, GetData()).WillRepeatedly(Return(data_view));
  webrtc::VideoFrameMetadata metadata;
  std::vector<uint32_t> csrcs{123, 321};
  // Copy csrcs rather than moving so we can compare in an EXPECT_EQ later.
  metadata.SetCsrcs(csrcs);

  EXPECT_CALL(original_frame, Metadata()).WillRepeatedly(Return(metadata));
  auto cloned_frame = CloneVideoFrame(&original_frame);

  EXPECT_EQ(cloned_frame->GetData().size(), 10u);
  EXPECT_THAT(cloned_frame->GetData(), Each(5u));
  EXPECT_EQ(cloned_frame->Metadata().GetCsrcs(), csrcs);
}

}  // namespace
}  // namespace webrtc
