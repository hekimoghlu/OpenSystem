/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 14, 2022.
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
#include <stdio.h>

#include <memory>

#include "rtc_base/checks.h"
#include "test/rtp_file_reader.h"
#include "test/rtp_file_writer.h"

using webrtc::test::RtpFileReader;
using webrtc::test::RtpFileWriter;

int main(int argc, char* argv[]) {
  if (argc < 3) {
    printf("Concatenates multiple rtpdump files into one.\n");
    printf("Usage: rtpcat in1.rtp int2.rtp [...] out.rtp\n");
    exit(1);
  }

  std::unique_ptr<RtpFileWriter> output(
      RtpFileWriter::Create(RtpFileWriter::kRtpDump, argv[argc - 1]));
  RTC_CHECK(output.get() != NULL) << "Cannot open output file.";
  printf("Output RTP file: %s\n", argv[argc - 1]);

  for (int i = 1; i < argc - 1; i++) {
    std::unique_ptr<RtpFileReader> input(
        RtpFileReader::Create(RtpFileReader::kRtpDump, argv[i]));
    RTC_CHECK(input.get() != NULL) << "Cannot open input file " << argv[i];
    printf("Input RTP file: %s\n", argv[i]);

    webrtc::test::RtpPacket packet;
    while (input->NextPacket(&packet))
      RTC_CHECK(output->WritePacket(&packet));
  }
  return 0;
}
