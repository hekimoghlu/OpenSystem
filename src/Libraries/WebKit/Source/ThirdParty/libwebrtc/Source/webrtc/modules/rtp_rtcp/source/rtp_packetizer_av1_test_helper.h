/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 17, 2023.
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
#ifndef MODULES_RTP_RTCP_SOURCE_RTP_PACKETIZER_AV1_TEST_HELPER_H_
#define MODULES_RTP_RTCP_SOURCE_RTP_PACKETIZER_AV1_TEST_HELPER_H_

#include <stdint.h>

#include <initializer_list>
#include <utility>
#include <vector>

namespace webrtc {
// All obu types offset by 3 to take correct position in the obu_header.
constexpr uint8_t kAv1ObuTypeSequenceHeader = 1 << 3;
constexpr uint8_t kAv1ObuTypeTemporalDelimiter = 2 << 3;
constexpr uint8_t kAv1ObuTypeFrameHeader = 3 << 3;
constexpr uint8_t kAv1ObuTypeTileGroup = 4 << 3;
constexpr uint8_t kAv1ObuTypeMetadata = 5 << 3;
constexpr uint8_t kAv1ObuTypeFrame = 6 << 3;
constexpr uint8_t kAv1ObuTypeTileList = 8 << 3;
constexpr uint8_t kAv1ObuExtensionPresentBit = 0b0'0000'100;
constexpr uint8_t kAv1ObuSizePresentBit = 0b0'0000'010;
constexpr uint8_t kAv1ObuExtensionS1T1 = 0b001'01'000;

class Av1Obu {
 public:
  explicit Av1Obu(uint8_t obu_type);

  Av1Obu& WithExtension(uint8_t extension);
  Av1Obu& WithoutSize();
  Av1Obu& WithPayload(std::vector<uint8_t> payload);

 private:
  friend std::vector<uint8_t> BuildAv1Frame(std::initializer_list<Av1Obu> obus);
  uint8_t header_;
  uint8_t extension_ = 0;
  std::vector<uint8_t> payload_;
};

std::vector<uint8_t> BuildAv1Frame(std::initializer_list<Av1Obu> obus);

}  // namespace webrtc
#endif  // MODULES_RTP_RTCP_SOURCE_RTP_PACKETIZER_AV1_TEST_HELPER_H_
