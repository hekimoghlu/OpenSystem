/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 11, 2023.
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
#ifndef TEST_FAKE_ENCODED_FRAME_H_
#define TEST_FAKE_ENCODED_FRAME_H_

#include <memory>
#include <vector>

#include "api/rtp_packet_infos.h"
#include "api/video/encoded_frame.h"
#include "api/video/video_rotation.h"
#include "test/gmock.h"

namespace webrtc {
namespace test {

class FakeEncodedFrame : public EncodedFrame {
 public:
  // Always 10ms delay and on time.
  int64_t ReceivedTime() const override;
  int64_t RenderTime() const override;

  // Setters for protected variables.
  void SetReceivedTime(int64_t received_time);
  void SetPayloadType(int payload_type);

 private:
  int64_t received_time_;
};

MATCHER_P(WithId, id, "") {
  return ::testing::Matches(::testing::Eq(id))(arg.Id());
}

MATCHER_P(FrameWithSize, id, "") {
  return ::testing::Matches(::testing::Eq(id))(arg.size());
}

MATCHER_P(RtpTimestamp, ts, "") {
  return ts == arg.RtpTimestamp();
}

class FakeFrameBuilder {
 public:
  FakeFrameBuilder& Time(uint32_t rtp_timestamp);
  FakeFrameBuilder& Id(int64_t frame_id);
  FakeFrameBuilder& AsLast();
  FakeFrameBuilder& Refs(const std::vector<int64_t>& references);
  FakeFrameBuilder& PlayoutDelay(VideoPlayoutDelay playout_delay);
  FakeFrameBuilder& SpatialLayer(int spatial_layer);
  FakeFrameBuilder& ReceivedTime(Timestamp receive_time);
  FakeFrameBuilder& Size(size_t size);
  FakeFrameBuilder& PayloadType(int payload_type);
  FakeFrameBuilder& NtpTime(Timestamp ntp_time);
  FakeFrameBuilder& Rotation(VideoRotation rotation);
  FakeFrameBuilder& PacketInfos(RtpPacketInfos packet_infos);
  std::unique_ptr<FakeEncodedFrame> Build();

 private:
  std::optional<uint32_t> rtp_timestamp_;
  std::optional<int64_t> frame_id_;
  std::optional<VideoPlayoutDelay> playout_delay_;
  std::optional<int> spatial_layer_;
  std::optional<Timestamp> received_time_;
  std::optional<int> payload_type_;
  std::optional<Timestamp> ntp_time_;
  std::optional<VideoRotation> rotation_;
  std::optional<RtpPacketInfos> packet_infos_;
  std::vector<int64_t> references_;
  bool last_spatial_layer_ = false;
  size_t size_ = 10;
};

}  // namespace test
}  // namespace webrtc

#endif  // TEST_FAKE_ENCODED_FRAME_H_
