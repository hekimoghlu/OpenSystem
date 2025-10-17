/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 8, 2022.
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
#ifndef RTC_TOOLS_NETWORK_TESTER_CONFIG_READER_H_
#define RTC_TOOLS_NETWORK_TESTER_CONFIG_READER_H_

#include <fstream>
#include <optional>
#include <string>

#ifdef WEBRTC_NETWORK_TESTER_PROTO
#include "rtc_tools/network_tester/network_tester_config.pb.h"
using webrtc::network_tester::config::NetworkTesterAllConfigs;
#else
class NetworkTesterConfigs;
#endif  // WEBRTC_NETWORK_TESTER_PROTO

namespace webrtc {

class ConfigReader {
 public:
  struct Config {
    int packet_send_interval_ms;
    int packet_size;
    int execution_time_ms;
  };
  explicit ConfigReader(const std::string& config_file_path);
  ~ConfigReader();

  ConfigReader(const ConfigReader&) = delete;
  ConfigReader& operator=(const ConfigReader&) = delete;

  std::optional<Config> GetNextConfig();

 private:
  NetworkTesterAllConfigs proto_all_configs_;
  int proto_config_index_;
};

}  // namespace webrtc

#endif  // RTC_TOOLS_NETWORK_TESTER_CONFIG_READER_H_
