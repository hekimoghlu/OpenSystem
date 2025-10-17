/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 14, 2023.
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
#include "modules/remote_bitrate_estimator/tools/bwe_rtp.h"

#include <stdio.h>

#include <set>
#include <sstream>
#include <string>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "modules/rtp_rtcp/include/rtp_header_extension_map.h"
#include "test/rtp_file_reader.h"

ABSL_FLAG(std::string,
          extension_type,
          "abs",
          "Extension type, either abs for absolute send time or tsoffset "
          "for timestamp offset.");
std::string ExtensionType() {
  return absl::GetFlag(FLAGS_extension_type);
}

ABSL_FLAG(int, extension_id, 3, "Extension id.");
int ExtensionId() {
  return absl::GetFlag(FLAGS_extension_id);
}

ABSL_FLAG(std::string, input_file, "", "Input file.");
std::string InputFile() {
  return absl::GetFlag(FLAGS_input_file);
}

ABSL_FLAG(std::string,
          ssrc_filter,
          "",
          "Comma-separated list of SSRCs in hexadecimal which are to be "
          "used as input to the BWE (only applicable to pcap files).");
std::set<uint32_t> SsrcFilter() {
  std::string ssrc_filter_string = absl::GetFlag(FLAGS_ssrc_filter);
  if (ssrc_filter_string.empty())
    return std::set<uint32_t>();
  std::stringstream ss;
  std::string ssrc_filter = ssrc_filter_string;
  std::set<uint32_t> ssrcs;

  // Parse the ssrcs in hexadecimal format.
  ss << std::hex << ssrc_filter;
  uint32_t ssrc;
  while (ss >> ssrc) {
    ssrcs.insert(ssrc);
    ss.ignore(1, ',');
  }
  return ssrcs;
}

bool ParseArgsAndSetupRtpReader(
    int argc,
    char** argv,
    std::unique_ptr<webrtc::test::RtpFileReader>& rtp_reader,
    webrtc::RtpHeaderExtensionMap& rtp_header_extensions) {
  absl::ParseCommandLine(argc, argv);
  std::string filename = InputFile();

  std::set<uint32_t> ssrc_filter = SsrcFilter();
  fprintf(stderr, "Filter on SSRC: ");
  for (auto& s : ssrc_filter) {
    fprintf(stderr, "0x%08x, ", s);
  }
  fprintf(stderr, "\n");
  if (filename.substr(filename.find_last_of('.')) == ".pcap") {
    fprintf(stderr, "Opening as pcap\n");
    rtp_reader.reset(webrtc::test::RtpFileReader::Create(
        webrtc::test::RtpFileReader::kPcap, filename.c_str(), SsrcFilter()));
  } else {
    fprintf(stderr, "Opening as rtp\n");
    rtp_reader.reset(webrtc::test::RtpFileReader::Create(
        webrtc::test::RtpFileReader::kRtpDump, filename.c_str()));
  }
  if (!rtp_reader) {
    fprintf(stderr, "Cannot open input file %s\n", filename.c_str());
    return false;
  }
  fprintf(stderr, "Input file: %s\n\n", filename.c_str());

  webrtc::RTPExtensionType extension = webrtc::kRtpExtensionAbsoluteSendTime;
  if (ExtensionType() == "tsoffset") {
    extension = webrtc::kRtpExtensionTransmissionTimeOffset;
    fprintf(stderr, "Extension: toffset\n");
  } else if (ExtensionType() == "abs") {
    fprintf(stderr, "Extension: abs\n");
  } else {
    fprintf(stderr, "Unknown extension type\n");
    return false;
  }

  rtp_header_extensions.RegisterByType(ExtensionId(), extension);

  return true;
}
