/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 31, 2025.
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
#ifndef MODULES_RTP_RTCP_SOURCE_RTCP_PACKET_RRTR_H_
#define MODULES_RTP_RTCP_SOURCE_RTCP_PACKET_RRTR_H_

#include <stddef.h>
#include <stdint.h>

#include "system_wrappers/include/ntp_time.h"

namespace webrtc {
namespace rtcp {

class Rrtr {
 public:
  static constexpr uint8_t kBlockType = 4;
  static constexpr uint16_t kBlockLength = 2;
  static constexpr size_t kLength = 4 * (kBlockLength + 1);  // 12

  Rrtr() {}
  Rrtr(const Rrtr&) = default;
  ~Rrtr() {}

  Rrtr& operator=(const Rrtr&) = default;

  void Parse(const uint8_t* buffer);

  // Fills buffer with the Rrtr.
  // Consumes Rrtr::kLength bytes.
  void Create(uint8_t* buffer) const;

  void SetNtp(NtpTime ntp) { ntp_ = ntp; }

  NtpTime ntp() const { return ntp_; }

 private:
  NtpTime ntp_;
};

inline bool operator==(const Rrtr& rrtr1, const Rrtr& rrtr2) {
  return rrtr1.ntp() == rrtr2.ntp();
}

inline bool operator!=(const Rrtr& rrtr1, const Rrtr& rrtr2) {
  return !(rrtr1 == rrtr2);
}

}  // namespace rtcp
}  // namespace webrtc
#endif  // MODULES_RTP_RTCP_SOURCE_RTCP_PACKET_RRTR_H_
