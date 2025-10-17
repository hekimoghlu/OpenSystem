/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 31, 2025.
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
#include "test/fake_encoded_frame.h"

#include <memory>

#include "api/video/video_frame_type.h"

namespace webrtc {
namespace test {

int64_t FakeEncodedFrame::ReceivedTime() const {
  return received_time_;
}

int64_t FakeEncodedFrame::RenderTime() const {
  return _renderTimeMs;
}

void FakeEncodedFrame::SetReceivedTime(int64_t received_time) {
  received_time_ = received_time;
}

void FakeEncodedFrame::SetPayloadType(int payload_type) {
  _payloadType = payload_type;
}

FakeFrameBuilder& FakeFrameBuilder::Time(uint32_t rtp_timestamp) {
  rtp_timestamp_ = rtp_timestamp;
  return *this;
}

FakeFrameBuilder& FakeFrameBuilder::Id(int64_t frame_id) {
  frame_id_ = frame_id;
  return *this;
}

FakeFrameBuilder& FakeFrameBuilder::AsLast() {
  last_spatial_layer_ = true;
  return *this;
}

FakeFrameBuilder& FakeFrameBuilder::Refs(
    const std::vector<int64_t>& references) {
  references_ = references;
  return *this;
}

FakeFrameBuilder& FakeFrameBuilder::PlayoutDelay(
    VideoPlayoutDelay playout_delay) {
  playout_delay_ = playout_delay;
  return *this;
}

FakeFrameBuilder& FakeFrameBuilder::SpatialLayer(int spatial_layer) {
  spatial_layer_ = spatial_layer;
  return *this;
}

FakeFrameBuilder& FakeFrameBuilder::ReceivedTime(Timestamp receive_time) {
  received_time_ = receive_time;
  return *this;
}

FakeFrameBuilder& FakeFrameBuilder::Size(size_t size) {
  size_ = size;
  return *this;
}

std::unique_ptr<FakeEncodedFrame> FakeFrameBuilder::Build() {
  RTC_CHECK_LE(references_.size(), EncodedFrame::kMaxFrameReferences);

  auto frame = std::make_unique<FakeEncodedFrame>();
  frame->is_last_spatial_layer = last_spatial_layer_;
  frame->SetEncodedData(EncodedImageBuffer::Create(size_));

  if (rtp_timestamp_)
    frame->SetRtpTimestamp(*rtp_timestamp_);
  if (frame_id_)
    frame->SetId(*frame_id_);
  if (playout_delay_)
    frame->SetPlayoutDelay(*playout_delay_);
  frame->SetFrameType(references_.empty() ? VideoFrameType::kVideoFrameKey
                                          : VideoFrameType::kVideoFrameDelta);
  for (int64_t ref : references_) {
    frame->references[frame->num_references] = ref;
    frame->num_references++;
  }
  if (spatial_layer_)
    frame->SetSpatialIndex(spatial_layer_);
  if (received_time_)
    frame->SetReceivedTime(received_time_->ms());
  if (payload_type_)
    frame->SetPayloadType(*payload_type_);
  if (ntp_time_)
    frame->ntp_time_ms_ = ntp_time_->ms();
  if (rotation_)
    frame->rotation_ = *rotation_;
  if (packet_infos_)
    frame->SetPacketInfos(*packet_infos_);
  return frame;
}

FakeFrameBuilder& FakeFrameBuilder::PayloadType(int payload_type) {
  payload_type_ = payload_type;
  return *this;
}

FakeFrameBuilder& FakeFrameBuilder::NtpTime(Timestamp ntp_time) {
  ntp_time_ = ntp_time;
  return *this;
}

FakeFrameBuilder& FakeFrameBuilder::Rotation(VideoRotation rotation) {
  rotation_ = rotation;
  return *this;
}

FakeFrameBuilder& FakeFrameBuilder::PacketInfos(RtpPacketInfos packet_infos) {
  packet_infos_ = packet_infos;
  return *this;
}

}  // namespace test
}  // namespace webrtc
