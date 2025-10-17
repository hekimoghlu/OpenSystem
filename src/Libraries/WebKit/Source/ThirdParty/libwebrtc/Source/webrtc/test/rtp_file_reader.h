/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 8, 2022.
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
#ifndef TEST_RTP_FILE_READER_H_
#define TEST_RTP_FILE_READER_H_

#include <cstdint>
#include <set>
#include <string>

#include "absl/strings/string_view.h"

namespace webrtc {
namespace test {

struct RtpPacket {
  // Accommodate for 50 ms packets of 32 kHz PCM16 samples (3200 bytes) plus
  // some overhead.
  static const size_t kMaxPacketBufferSize = 3500;
  uint8_t data[kMaxPacketBufferSize];
  size_t length;
  // The length the packet had on wire. Will be different from `length` when
  // reading a header-only RTP dump.
  size_t original_length;

  uint32_t time_ms;
};

class RtpFileReader {
 public:
  enum FileFormat { kPcap, kRtpDump, kLengthPacketInterleaved };

  virtual ~RtpFileReader() {}
  static RtpFileReader* Create(FileFormat format,
                               const uint8_t* data,
                               size_t size,
                               const std::set<uint32_t>& ssrc_filter);
  static RtpFileReader* Create(FileFormat format, absl::string_view filename);
  static RtpFileReader* Create(FileFormat format,
                               absl::string_view filename,
                               const std::set<uint32_t>& ssrc_filter);
  virtual bool NextPacket(RtpPacket* packet) = 0;
};
}  // namespace test
}  // namespace webrtc
#endif  // TEST_RTP_FILE_READER_H_
