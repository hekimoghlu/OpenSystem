/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 21, 2022.
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

#include <string>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/flags/usage.h"
#include "absl/strings/string_view.h"
#include "logging/rtc_event_log/rtc_event_log_parser.h"
#include "rtc_base/logging.h"
#include "rtc_tools/rtc_event_log_to_text/converter.h"

ABSL_FLAG(bool,
          parse_unconfigured_header_extensions,
          true,
          "Attempt to parse unconfigured header extensions using the default "
          "WebRTC mapping. This can give very misleading results if the "
          "application negotiates a different mapping.");

// Prints an RTC event log as human readable text with one line per event.
// Note that the RTC event log text format isn't an API. Prefer to build
// tools directly on the parser (logging/rtc_event_log/rtc_event_log_parser.cc).
int main(int argc, char* argv[]) {
  absl::SetProgramUsageMessage(
      "A tool for converting WebRTC event logs to text.\n"
      "The events are sorted by log time and printed\n"
      "with one event per line, using the following format:\n"
      "<EVENT_TYPE> <log_time_ms> <field1>=<value1> <field2>=<value2> ...\n"
      "\n"
      "Example usage:\n"
      "./rtc_event_log_to_text <inputfile> <outputfile>\n"
      "./rtc_event_log_to_text <inputfile>\n"
      "If no output file is specified, the output is written to stdout\n");
  std::vector<char*> args = absl::ParseCommandLine(argc, argv);

  // Print RTC_LOG warnings and errors even in release builds.
  if (rtc::LogMessage::GetLogToDebug() > rtc::LS_WARNING) {
    rtc::LogMessage::LogToDebug(rtc::LS_WARNING);
  }
  rtc::LogMessage::SetLogToStderr(true);

  webrtc::ParsedRtcEventLog::UnconfiguredHeaderExtensions header_extensions =
      webrtc::ParsedRtcEventLog::UnconfiguredHeaderExtensions::kDontParse;
  if (absl::GetFlag(FLAGS_parse_unconfigured_header_extensions)) {
    header_extensions = webrtc::ParsedRtcEventLog::
        UnconfiguredHeaderExtensions::kAttemptWebrtcDefaultConfig;
  }

  std::string inputfile;
  FILE* output = nullptr;
  if (args.size() == 3) {
    inputfile = args[1];
    std::string outputfile = args[2];
    output = fopen(outputfile.c_str(), "w");
  } else if (args.size() == 2) {
    inputfile = args[1];
    output = stdout;
  } else {
    // Print usage information.
    absl::string_view usage = absl::ProgramUsageMessage();
    fwrite(usage.data(), usage.size(), 1, stderr);
    return 1;
  }

  bool success = webrtc::Convert(inputfile, output, header_extensions);

  return success ? 0 : 1;
}
