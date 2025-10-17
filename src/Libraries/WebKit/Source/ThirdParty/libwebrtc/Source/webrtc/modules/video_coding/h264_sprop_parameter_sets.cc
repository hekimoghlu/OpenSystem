/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 17, 2021.
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
#include "modules/video_coding/h264_sprop_parameter_sets.h"

#include <stddef.h>
#include <stdint.h>

#include <string>
#include <vector>

#include "rtc_base/logging.h"
#include "rtc_base/third_party/base64/base64.h"

namespace {

bool DecodeAndConvert(const std::string& base64, std::vector<uint8_t>* binary) {
  return rtc::Base64::DecodeFromArray(base64.data(), base64.size(),
                                      rtc::Base64::DO_STRICT, binary, nullptr);
}
}  // namespace

namespace webrtc {

bool H264SpropParameterSets::DecodeSprop(const std::string& sprop) {
  size_t separator_pos = sprop.find(',');
  RTC_LOG(LS_INFO) << "Parsing sprop \"" << sprop << "\"";
  if ((separator_pos <= 0) || (separator_pos >= sprop.length() - 1)) {
    RTC_LOG(LS_WARNING) << "Invalid seperator position " << separator_pos
                        << " *" << sprop << "*";
    return false;
  }
  std::string sps_str = sprop.substr(0, separator_pos);
  std::string pps_str = sprop.substr(separator_pos + 1, std::string::npos);
  if (!DecodeAndConvert(sps_str, &sps_)) {
    RTC_LOG(LS_WARNING) << "Failed to decode sprop/sps *" << sprop << "*";
    return false;
  }
  if (!DecodeAndConvert(pps_str, &pps_)) {
    RTC_LOG(LS_WARNING) << "Failed to decode sprop/pps *" << sprop << "*";
    return false;
  }
  return true;
}

}  // namespace webrtc
