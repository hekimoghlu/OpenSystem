/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 13, 2023.
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
#ifndef STATS_TEST_RTC_TEST_STATS_H_
#define STATS_TEST_RTC_TEST_STATS_H_

#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <vector>

#include "api/stats/rtc_stats.h"
#include "rtc_base/system/rtc_export.h"

namespace webrtc {

class RTC_EXPORT RTCTestStats : public RTCStats {
 public:
  WEBRTC_RTCSTATS_DECL();
  RTCTestStats(const std::string& id, Timestamp timestamp);
  ~RTCTestStats() override;

  std::optional<bool> m_bool;
  std::optional<int32_t> m_int32;
  std::optional<uint32_t> m_uint32;
  std::optional<int64_t> m_int64;
  std::optional<uint64_t> m_uint64;
  std::optional<double> m_double;
  std::optional<std::string> m_string;
  std::optional<std::vector<bool>> m_sequence_bool;
  std::optional<std::vector<int32_t>> m_sequence_int32;
  std::optional<std::vector<uint32_t>> m_sequence_uint32;
  std::optional<std::vector<int64_t>> m_sequence_int64;
  std::optional<std::vector<uint64_t>> m_sequence_uint64;
  std::optional<std::vector<double>> m_sequence_double;
  std::optional<std::vector<std::string>> m_sequence_string;
  std::optional<std::map<std::string, uint64_t>> m_map_string_uint64;
  std::optional<std::map<std::string, double>> m_map_string_double;
};

}  // namespace webrtc

#endif  // STATS_TEST_RTC_TEST_STATS_H_
