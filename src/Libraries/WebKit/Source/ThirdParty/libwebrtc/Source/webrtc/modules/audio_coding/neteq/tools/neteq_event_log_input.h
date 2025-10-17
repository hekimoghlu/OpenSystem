/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 31, 2024.
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
#ifndef MODULES_AUDIO_CODING_NETEQ_TOOLS_NETEQ_EVENT_LOG_INPUT_H_
#define MODULES_AUDIO_CODING_NETEQ_TOOLS_NETEQ_EVENT_LOG_INPUT_H_

#include <optional>

#include "absl/strings/string_view.h"
#include "logging/rtc_event_log/rtc_event_log_parser.h"
#include "modules/audio_coding/neteq/tools/neteq_input.h"

namespace webrtc {
namespace test {

std::unique_ptr<NetEqInput> CreateNetEqEventLogInput(
    const ParsedRtcEventLog& parsed_log,
    std::optional<uint32_t> ssrc);

}  // namespace test
}  // namespace webrtc
#endif  // MODULES_AUDIO_CODING_NETEQ_TOOLS_NETEQ_EVENT_LOG_INPUT_H_
