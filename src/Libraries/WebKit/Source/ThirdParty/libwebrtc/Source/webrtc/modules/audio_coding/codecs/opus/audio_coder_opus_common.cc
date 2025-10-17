/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 24, 2023.
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
#include "modules/audio_coding/codecs/opus/audio_coder_opus_common.h"

#include "absl/strings/string_view.h"

namespace webrtc {

std::optional<std::string> GetFormatParameter(const SdpAudioFormat& format,
                                              absl::string_view param) {
  auto it = format.parameters.find(std::string(param));
  if (it == format.parameters.end())
    return std::nullopt;

  return it->second;
}

// Parses a comma-separated string "1,2,0,6" into a std::vector<unsigned char>.
template <>
std::optional<std::vector<unsigned char>> GetFormatParameter(
    const SdpAudioFormat& format,
    absl::string_view param) {
  std::vector<unsigned char> result;
  const std::string comma_separated_list =
      GetFormatParameter(format, param).value_or("");
  size_t pos = 0;
  while (pos < comma_separated_list.size()) {
    const size_t next_comma = comma_separated_list.find(',', pos);
    const size_t distance_to_next_comma = next_comma == std::string::npos
                                              ? std::string::npos
                                              : (next_comma - pos);
    auto substring_with_number =
        comma_separated_list.substr(pos, distance_to_next_comma);
    auto conv = rtc::StringToNumber<int>(substring_with_number);
    if (!conv.has_value()) {
      return std::nullopt;
    }
    result.push_back(*conv);
    pos += substring_with_number.size() + 1;
  }
  return result;
}

}  // namespace webrtc
