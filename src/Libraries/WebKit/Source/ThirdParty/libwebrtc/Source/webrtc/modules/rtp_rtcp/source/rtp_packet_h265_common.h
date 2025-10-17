/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 18, 2022.
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
#ifndef MODULES_RTP_RTCP_SOURCE_RTP_PACKET_H265_COMMON_H_
#define MODULES_RTP_RTCP_SOURCE_RTP_PACKET_H265_COMMON_H_

#include <string>
#include <vector>

namespace webrtc {
// The payload header consists of the same
// fields (F, Type, LayerId and TID) as the NAL unit header. Refer to
// section 4.4 in RFC 7798.
constexpr size_t kH265PayloadHeaderSizeBytes = 2;
constexpr uint8_t kH265MaxLayerId = 127;
constexpr uint8_t kH265MaxTemporalId = 7;
// Unlike H.264, H.265 NAL header is 2-bytes.
constexpr size_t kH265NalHeaderSizeBytes = 2;
// H.265's FU is constructed of 2-byte payload header, 1-byte FU header and FU
// payload.
constexpr size_t kH265FuHeaderSizeBytes = 1;
// The NALU size for H.265 RTP aggregated packet indicates the size of the NAL
// unit is 2-bytes.
constexpr size_t kH265LengthFieldSizeBytes = 2;
constexpr size_t kH265ApHeaderSizeBytes =
    kH265NalHeaderSizeBytes + kH265LengthFieldSizeBytes;

// Bit masks for NAL headers.
enum NalHdrMasks {
  kH265FBit = 0x80,
  kH265TypeMask = 0x7E,
  kH265LayerIDHMask = 0x1,
  kH265LayerIDLMask = 0xF8,
  kH265TIDMask = 0x7,
  kH265TypeMaskN = 0x81,
  kH265TypeMaskInFuHeader = 0x3F
};

// Bit masks for FU headers.
enum FuBitmasks {
  kH265SBitMask = 0x80,
  kH265EBitMask = 0x40,
  kH265FuTypeBitMask = 0x3F
};

constexpr uint8_t kStartCode[] = {0, 0, 0, 1};

}  // namespace webrtc

#endif  // MODULES_RTP_RTCP_SOURCE_RTP_PACKET_H265_COMMON_H_
