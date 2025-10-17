/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 11, 2022.
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
#include "modules/video_coding/deprecated/stream_generator.h"

#include <string.h>

#include <list>

#include "modules/video_coding/deprecated/packet.h"
#include "rtc_base/checks.h"

namespace webrtc {

StreamGenerator::StreamGenerator(uint16_t start_seq_num, int64_t current_time)
    : packets_(), sequence_number_(start_seq_num), start_time_(current_time) {}

void StreamGenerator::Init(uint16_t start_seq_num, int64_t current_time) {
  packets_.clear();
  sequence_number_ = start_seq_num;
  start_time_ = current_time;
  memset(packet_buffer_, 0, sizeof(packet_buffer_));
}

void StreamGenerator::GenerateFrame(VideoFrameType type,
                                    int num_media_packets,
                                    int num_empty_packets,
                                    int64_t time_ms) {
  uint32_t timestamp = 90 * (time_ms - start_time_);
  for (int i = 0; i < num_media_packets; ++i) {
    const int packet_size =
        (kFrameSize + num_media_packets / 2) / num_media_packets;
    bool marker_bit = (i == num_media_packets - 1);
    packets_.push_back(GeneratePacket(sequence_number_, timestamp, packet_size,
                                      (i == 0), marker_bit, type));
    ++sequence_number_;
  }
  for (int i = 0; i < num_empty_packets; ++i) {
    packets_.push_back(GeneratePacket(sequence_number_, timestamp, 0, false,
                                      false, VideoFrameType::kEmptyFrame));
    ++sequence_number_;
  }
}

VCMPacket StreamGenerator::GeneratePacket(uint16_t sequence_number,
                                          uint32_t timestamp,
                                          unsigned int size,
                                          bool first_packet,
                                          bool marker_bit,
                                          VideoFrameType type) {
  RTC_CHECK_LT(size, kMaxPacketSize);
  VCMPacket packet;
  packet.seqNum = sequence_number;
  packet.timestamp = timestamp;
  packet.video_header.frame_type = type;
  packet.video_header.is_first_packet_in_frame = first_packet;
  packet.markerBit = marker_bit;
  packet.sizeBytes = size;
  packet.dataPtr = packet_buffer_;
  if (packet.is_first_packet_in_frame())
    packet.completeNALU = kNaluStart;
  else if (packet.markerBit)
    packet.completeNALU = kNaluEnd;
  else
    packet.completeNALU = kNaluIncomplete;
  return packet;
}

bool StreamGenerator::PopPacket(VCMPacket* packet, int index) {
  std::list<VCMPacket>::iterator it = GetPacketIterator(index);
  if (it == packets_.end())
    return false;
  if (packet)
    *packet = (*it);
  packets_.erase(it);
  return true;
}

bool StreamGenerator::GetPacket(VCMPacket* packet, int index) {
  std::list<VCMPacket>::iterator it = GetPacketIterator(index);
  if (it == packets_.end())
    return false;
  if (packet)
    *packet = (*it);
  return true;
}

bool StreamGenerator::NextPacket(VCMPacket* packet) {
  if (packets_.empty())
    return false;
  if (packet != NULL)
    *packet = packets_.front();
  packets_.pop_front();
  return true;
}

void StreamGenerator::DropLastPacket() {
  packets_.pop_back();
}

uint16_t StreamGenerator::NextSequenceNumber() const {
  if (packets_.empty())
    return sequence_number_;
  return packets_.front().seqNum;
}

int StreamGenerator::PacketsRemaining() const {
  return packets_.size();
}

std::list<VCMPacket>::iterator StreamGenerator::GetPacketIterator(int index) {
  std::list<VCMPacket>::iterator it = packets_.begin();
  for (int i = 0; i < index; ++i) {
    ++it;
    if (it == packets_.end())
      break;
  }
  return it;
}

}  // namespace webrtc
