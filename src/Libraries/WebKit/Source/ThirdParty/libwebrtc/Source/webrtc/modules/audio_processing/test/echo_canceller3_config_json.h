/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 14, 2025.
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
#ifndef MODULES_AUDIO_PROCESSING_TEST_ECHO_CANCELLER3_CONFIG_JSON_H_
#define MODULES_AUDIO_PROCESSING_TEST_ECHO_CANCELLER3_CONFIG_JSON_H_

#include <string>

#include "absl/strings/string_view.h"
#include "api/audio/echo_canceller3_config.h"

namespace webrtc {
// Parses a JSON-encoded string into an Aec3 config. Fields corresponds to
// substruct names, with the addition that there must be a top-level node
// "aec3". Produces default config values for anything that cannot be parsed
// from the string. If any error was found in the parsing, parsing_successful is
// set to false.
void Aec3ConfigFromJsonString(absl::string_view json_string,
                                         EchoCanceller3Config* config,
                                         bool* parsing_successful);

// Encodes an Aec3 config in JSON format. Fields corresponds to substruct names,
// with the addition that the top-level node is named "aec3".
std::string Aec3ConfigToJsonString(
    const EchoCanceller3Config& config);

}  // namespace webrtc

#endif  // MODULES_AUDIO_PROCESSING_TEST_ECHO_CANCELLER3_CONFIG_JSON_H_
