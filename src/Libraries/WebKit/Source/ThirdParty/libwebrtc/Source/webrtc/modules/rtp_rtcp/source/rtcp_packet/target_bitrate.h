/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 27, 2023.
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
#ifndef MODULES_RTP_RTCP_SOURCE_RTCP_PACKET_TARGET_BITRATE_H_
#define MODULES_RTP_RTCP_SOURCE_RTCP_PACKET_TARGET_BITRATE_H_

#include <stddef.h>
#include <stdint.h>

#include <vector>

namespace webrtc {
namespace rtcp {

class TargetBitrate {
 public:
  // TODO(sprang): This block type is just a place holder. We need to get an
  //               id assigned by IANA.
  static constexpr uint8_t kBlockType = 42;
  static constexpr size_t kBitrateItemSizeBytes = 4;

  struct BitrateItem {
    BitrateItem();
    BitrateItem(uint8_t spatial_layer,
                uint8_t temporal_layer,
                uint32_t target_bitrate_kbps);

    uint8_t spatial_layer;
    uint8_t temporal_layer;
    uint32_t target_bitrate_kbps;
  };

  TargetBitrate();
  TargetBitrate(const TargetBitrate&);
  TargetBitrate& operator=(const TargetBitrate&);
  ~TargetBitrate();

  void AddTargetBitrate(uint8_t spatial_layer,
                        uint8_t temporal_layer,
                        uint32_t target_bitrate_kbps);

  const std::vector<BitrateItem>& GetTargetBitrates() const;

  void Parse(const uint8_t* block, uint16_t block_length);

  size_t BlockLength() const;

  void Create(uint8_t* buffer) const;

 private:
  std::vector<BitrateItem> bitrates_;
};

}  // namespace rtcp
}  // namespace webrtc
#endif  // MODULES_RTP_RTCP_SOURCE_RTCP_PACKET_TARGET_BITRATE_H_
