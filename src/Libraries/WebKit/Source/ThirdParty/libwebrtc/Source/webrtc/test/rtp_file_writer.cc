/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 27, 2023.
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
#include "test/rtp_file_writer.h"

#include <stdint.h>
#include <stdio.h>

#include <optional>
#include <string>

#include "rtc_base/checks.h"

namespace webrtc {
namespace test {

static const uint16_t kPacketHeaderSize = 8;
static const char kFirstLine[] = "#!rtpplay1.0 0.0.0.0/0\n";

// Write RTP packets to file in rtpdump format, as documented at:
// http://www.cs.columbia.edu/irt/software/rtptools/
class RtpDumpWriter : public RtpFileWriter {
 public:
  explicit RtpDumpWriter(FILE* file) : file_(file) {
    RTC_CHECK(file_ != NULL);
    Init();
  }
  ~RtpDumpWriter() override {
    if (file_ != NULL) {
      fclose(file_);
      file_ = NULL;
    }
  }

  RtpDumpWriter(const RtpDumpWriter&) = delete;
  RtpDumpWriter& operator=(const RtpDumpWriter&) = delete;

  bool WritePacket(const RtpPacket* packet) override {
    if (!first_packet_time_) {
      first_packet_time_ = packet->time_ms;
    }
    uint16_t len = static_cast<uint16_t>(packet->length + kPacketHeaderSize);
    uint16_t plen = static_cast<uint16_t>(packet->original_length);
    uint32_t offset = packet->time_ms - *first_packet_time_;
    RTC_CHECK(WriteUint16(len));
    RTC_CHECK(WriteUint16(plen));
    RTC_CHECK(WriteUint32(offset));
    return fwrite(packet->data, sizeof(uint8_t), packet->length, file_) ==
           packet->length;
  }

 private:
  bool Init() {
    fprintf(file_, "%s", kFirstLine);

    RTC_CHECK(WriteUint32(0));
    RTC_CHECK(WriteUint32(0));
    RTC_CHECK(WriteUint32(0));
    RTC_CHECK(WriteUint16(0));
    RTC_CHECK(WriteUint16(0));

    return true;
  }

  bool WriteUint32(uint32_t in) {
    // Loop through shifts = {24, 16, 8, 0}.
    for (int shifts = 24; shifts >= 0; shifts -= 8) {
      uint8_t tmp = static_cast<uint8_t>((in >> shifts) & 0xFF);
      if (fwrite(&tmp, sizeof(uint8_t), 1, file_) != 1)
        return false;
    }
    return true;
  }

  bool WriteUint16(uint16_t in) {
    // Write 8 MSBs.
    uint8_t tmp = static_cast<uint8_t>((in >> 8) & 0xFF);
    if (fwrite(&tmp, sizeof(uint8_t), 1, file_) != 1)
      return false;
    // Write 8 LSBs.
    tmp = static_cast<uint8_t>(in & 0xFF);
    if (fwrite(&tmp, sizeof(uint8_t), 1, file_) != 1)
      return false;
    return true;
  }

  FILE* file_;
  std::optional<uint32_t> first_packet_time_;
};

RtpFileWriter* RtpFileWriter::Create(FileFormat format,
                                     const std::string& filename) {
  FILE* file = fopen(filename.c_str(), "wb");
  if (file == NULL) {
    printf("ERROR: Can't open file: %s\n", filename.c_str());
    return NULL;
  }
  switch (format) {
    case kRtpDump:
      return new RtpDumpWriter(file);
  }
  fclose(file);
  return NULL;
}

}  // namespace test
}  // namespace webrtc
