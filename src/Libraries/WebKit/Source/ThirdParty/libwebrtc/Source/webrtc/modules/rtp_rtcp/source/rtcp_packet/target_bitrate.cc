/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 8, 2025.
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
#include "modules/rtp_rtcp/source/rtcp_packet/target_bitrate.h"

#include "modules/rtp_rtcp/source/byte_io.h"
#include "rtc_base/checks.h"
#include "rtc_base/numerics/safe_conversions.h"

namespace webrtc {
namespace rtcp {
namespace {

constexpr size_t kTargetBitrateHeaderSizeBytes = 4;

}  // namespace

TargetBitrate::BitrateItem::BitrateItem()
    : spatial_layer(0), temporal_layer(0), target_bitrate_kbps(0) {}

TargetBitrate::BitrateItem::BitrateItem(uint8_t spatial_layer,
                                        uint8_t temporal_layer,
                                        uint32_t target_bitrate_kbps)
    : spatial_layer(spatial_layer),
      temporal_layer(temporal_layer),
      target_bitrate_kbps(target_bitrate_kbps) {}

//  RFC 4585: Feedback format.
//
//  Common packet format:
//
//   0                   1                   2                   3
//   0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
//  +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
//  |     BT=42     |   reserved    |         block length          |
//  +=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+
//
//  Target bitrate item (repeat as many times as necessary).
//
//  +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
//  |   S   |   T   |                Target Bitrate                 |
//  +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
//  :  ...                                                          :
//
//  Spatial Layer (S): 4 bits
//    Indicates which temporal layer this bitrate concerns.
//
//  Temporal Layer (T): 4 bits
//    Indicates which temporal layer this bitrate concerns.
//
//  Target Bitrate: 24 bits
//    The encoder target bitrate for this layer, in kbps.
//
//  As an example of how S and T are intended to be used, VP8 simulcast will
//  use a separate TargetBitrate message per stream, since they are transmitted
//  on separate SSRCs, with temporal layers grouped by stream.
//  If VP9 SVC is used, there will be only one SSRC, so each spatial and
//  temporal layer combo used shall be specified in the TargetBitrate packet.

TargetBitrate::TargetBitrate() = default;
TargetBitrate::TargetBitrate(const TargetBitrate&) = default;
TargetBitrate& TargetBitrate::operator=(const TargetBitrate&) = default;
TargetBitrate::~TargetBitrate() = default;

void TargetBitrate::Parse(const uint8_t* block, uint16_t block_length) {
  // Validate block header (should already have been parsed and checked).
  RTC_DCHECK_EQ(block[0], kBlockType);
  RTC_DCHECK_EQ(block_length, ByteReader<uint16_t>::ReadBigEndian(&block[2]));

  // Header specifies block length - 1, but since we ignore the header, which
  // occupies exactly on block, we can just treat this as payload length.
  const size_t payload_bytes = block_length * 4;
  const size_t num_items = payload_bytes / kBitrateItemSizeBytes;
  size_t index = kTargetBitrateHeaderSizeBytes;
  bitrates_.clear();
  for (size_t i = 0; i < num_items; ++i) {
    uint8_t layers = block[index];
    uint32_t bitrate_kbps =
        ByteReader<uint32_t, 3>::ReadBigEndian(&block[index + 1]);
    index += kBitrateItemSizeBytes;
    AddTargetBitrate((layers >> 4) & 0x0F, layers & 0x0F, bitrate_kbps);
  }
}

void TargetBitrate::AddTargetBitrate(uint8_t spatial_layer,
                                     uint8_t temporal_layer,
                                     uint32_t target_bitrate_kbps) {
  RTC_DCHECK_LE(spatial_layer, 0x0F);
  RTC_DCHECK_LE(temporal_layer, 0x0F);
  RTC_DCHECK_LE(target_bitrate_kbps, 0x00FFFFFFU);
  bitrates_.push_back(
      BitrateItem(spatial_layer, temporal_layer, target_bitrate_kbps));
}

const std::vector<TargetBitrate::BitrateItem>&
TargetBitrate::GetTargetBitrates() const {
  return bitrates_;
}

size_t TargetBitrate::BlockLength() const {
  return kTargetBitrateHeaderSizeBytes +
         bitrates_.size() * kBitrateItemSizeBytes;
}

void TargetBitrate::Create(uint8_t* buffer) const {
  buffer[0] = kBlockType;
  buffer[1] = 0;  // Reserved.
  uint16_t block_length_words =
      rtc::dchecked_cast<uint16_t>((BlockLength() / 4) - 1);
  ByteWriter<uint16_t>::WriteBigEndian(&buffer[2], block_length_words);

  size_t index = kTargetBitrateHeaderSizeBytes;
  for (const BitrateItem& item : bitrates_) {
    buffer[index] = (item.spatial_layer << 4) | item.temporal_layer;
    ByteWriter<uint32_t, 3>::WriteBigEndian(&buffer[index + 1],
                                            item.target_bitrate_kbps);
    index += kBitrateItemSizeBytes;
  }
}

}  // namespace rtcp
}  // namespace webrtc
