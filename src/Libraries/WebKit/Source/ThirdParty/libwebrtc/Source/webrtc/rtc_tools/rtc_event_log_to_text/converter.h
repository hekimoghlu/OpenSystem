/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 16, 2023.
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
#ifndef RTC_TOOLS_RTC_EVENT_LOG_TO_TEXT_CONVERTER_H_
#define RTC_TOOLS_RTC_EVENT_LOG_TO_TEXT_CONVERTER_H_

#include <stdio.h>

#include <string>

#include "absl/base/attributes.h"
#include "logging/rtc_event_log/rtc_event_log_parser.h"

namespace webrtc {

// Parses events from file `inputfile` and prints human readable text
// representations to `output`. The output is sorted by log time.
// `handle_unconfigured_extensions` controls the policy for parsing RTP
// header extensions if the log doesn't contain a mapping between the
// header extensions and numerical IDs.
ABSL_MUST_USE_RESULT bool Convert(
    std::string inputfile,
    FILE* output,
    ParsedRtcEventLog::UnconfiguredHeaderExtensions
        handle_unconfigured_extensions);

}  // namespace webrtc

#endif  // RTC_TOOLS_RTC_EVENT_LOG_TO_TEXT_CONVERTER_H_
