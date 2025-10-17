/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 7, 2025.
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
#ifndef MODULES_VIDEO_CODING_PACKET_BUFFER_H_
#define MODULES_VIDEO_CODING_PACKET_BUFFER_H_

#include <memory>
#include <queue>
#include <set>
#include <vector>

#include "absl/base/attributes.h"
#include "api/rtp_packet_info.h"
#include "api/units/timestamp.h"
#include "api/video/encoded_image.h"
#include "modules/rtp_rtcp/source/rtp_packet_received.h"
#include "modules/rtp_rtcp/source/rtp_video_header.h"
#include "rtc_base/copy_on_write_buffer.h"
#include "rtc_base/numerics/sequence_number_util.h"
#include "rtc_base/thread_annotations.h"

namespace webrtc {
namespace video_coding {

class PacketBuffer {
 public:
  struct Packet {
    Packet() = default;
    Packet(const RtpPacketReceived& rtp_packet,
           int64_t sequence_number,
           const RTPVideoHeader& video_header);
    Packet(const Packet&) = delete;
    Packet(Packet&&) = delete;
    Packet& operator=(const Packet&) = delete;
    Packet& operator=(Packet&&) = delete;
    ~Packet() = default;

    VideoCodecType codec() const { return video_header.codec; }
    int width() const { return video_header.width; }
    int height() const { return video_header.height; }

    bool is_first_packet_in_frame() const {
      return video_header.is_first_packet_in_frame;
    }
    bool is_last_packet_in_frame() const {
      return video_header.is_last_packet_in_frame;
    }
    uint16_t seq_num() const { return static_cast<uint16_t>(sequence_number); }

    // If all its previous packets have been inserted into the packet buffer.
    // Set and used internally by the PacketBuffer.
    bool continuous = false;
    bool marker_bit = false;
    uint8_t payload_type = 0;
    int64_t sequence_number = 0;
    uint32_t timestamp = 0;
    int times_nacked = -1;

    rtc::CopyOnWriteBuffer video_payload;
    RTPVideoHeader video_header;
  };
  struct InsertResult {
    std::vector<std::unique_ptr<Packet>> packets;
    // Indicates if the packet buffer was cleared, which means that a key
    // frame request should be sent.
    bool buffer_cleared = false;
  };

  // Both `start_buffer_size` and `max_buffer_size` must be a power of 2.
  PacketBuffer(size_t start_buffer_size, size_t max_buffer_size);
  ~PacketBuffer();

  ABSL_MUST_USE_RESULT InsertResult
  InsertPacket(std::unique_ptr<Packet> packet);
  ABSL_MUST_USE_RESULT InsertResult InsertPadding(uint16_t seq_num);
  void ClearTo(uint16_t seq_num);
  void Clear();

  void ForceSpsPpsIdrIsH264Keyframe();
  void ResetSpsPpsIdrIsH264Keyframe();

 private:
  void ClearInternal();

  // Tries to expand the buffer.
  bool ExpandBufferSize();

  // Test if all previous packets has arrived for the given sequence number.
  bool PotentialNewFrame(uint16_t seq_num) const;

  // Test if all packets of a frame has arrived, and if so, returns packets to
  // create frames.
  std::vector<std::unique_ptr<Packet>> FindFrames(uint16_t seq_num);

  void UpdateMissingPackets(uint16_t seq_num);

  // buffer_.size() and max_size_ must always be a power of two.
  const size_t max_size_;

  // The fist sequence number currently in the buffer.
  uint16_t first_seq_num_;

  // If the packet buffer has received its first packet.
  bool first_packet_received_;

  // If the buffer is cleared to `first_seq_num_`.
  bool is_cleared_to_first_seq_num_;

  // Buffer that holds the the inserted packets and information needed to
  // determine continuity between them.
  std::vector<std::unique_ptr<Packet>> buffer_;

  std::optional<uint16_t> newest_inserted_seq_num_;
  std::set<uint16_t, DescendingSeqNumComp<uint16_t>> missing_packets_;

  std::set<uint16_t, DescendingSeqNumComp<uint16_t>> received_padding_;

  // Indicates if we should require SPS, PPS, and IDR for a particular
  // RTP timestamp to treat the corresponding frame as a keyframe.
  bool sps_pps_idr_is_h264_keyframe_;
};

}  // namespace video_coding
}  // namespace webrtc

#endif  // MODULES_VIDEO_CODING_PACKET_BUFFER_H_
