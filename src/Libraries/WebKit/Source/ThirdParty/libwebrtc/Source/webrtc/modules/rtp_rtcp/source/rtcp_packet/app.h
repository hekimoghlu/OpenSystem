/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 17, 2022.
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
#ifndef MODULES_RTP_RTCP_SOURCE_RTCP_PACKET_APP_H_
#define MODULES_RTP_RTCP_SOURCE_RTCP_PACKET_APP_H_

#include <stddef.h>
#include <stdint.h>

#include "modules/rtp_rtcp/source/rtcp_packet.h"
#include "rtc_base/buffer.h"

namespace webrtc {
namespace rtcp {
class CommonHeader;

class App : public RtcpPacket {
 public:
  static constexpr uint8_t kPacketType = 204;
  App();
  App(App&&) = default;
  ~App() override;

  // Parse assumes header is already parsed and validated.
  bool Parse(const CommonHeader& packet);

  void SetSubType(uint8_t subtype);
  void SetName(uint32_t name) { name_ = name; }
  void SetData(const uint8_t* data, size_t data_length);

  uint8_t sub_type() const { return sub_type_; }
  uint32_t name() const { return name_; }
  size_t data_size() const { return data_.size(); }
  const uint8_t* data() const { return data_.data(); }

  size_t BlockLength() const override;

  bool Create(uint8_t* packet,
              size_t* index,
              size_t max_length,
              PacketReadyCallback callback) const override;

  static inline constexpr uint32_t NameToInt(const char name[5]) {
    return static_cast<uint32_t>(name[0]) << 24 |
           static_cast<uint32_t>(name[1]) << 16 |
           static_cast<uint32_t>(name[2]) << 8 | static_cast<uint32_t>(name[3]);
  }

 private:
  static constexpr size_t kAppBaseLength = 8;  // Ssrc and Name.
  static constexpr size_t kMaxDataSize = 0xffff * 4 - kAppBaseLength;

  uint8_t sub_type_;
  uint32_t name_;
  rtc::Buffer data_;
};

}  // namespace rtcp
}  // namespace webrtc
#endif  // MODULES_RTP_RTCP_SOURCE_RTCP_PACKET_APP_H_
