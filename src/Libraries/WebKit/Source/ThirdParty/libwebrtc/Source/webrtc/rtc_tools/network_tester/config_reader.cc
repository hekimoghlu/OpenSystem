/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 27, 2024.
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
#include "rtc_tools/network_tester/config_reader.h"

#include <fstream>
#include <iterator>
#include <string>

#include "rtc_base/checks.h"

namespace webrtc {

ConfigReader::ConfigReader(const std::string& config_file_path)
    : proto_config_index_(0) {
  std::ifstream config_stream(config_file_path,
                              std::ios_base::in | std::ios_base::binary);
  RTC_DCHECK(config_stream.is_open())
      << "Config " << config_file_path << " open failed";
  RTC_DCHECK(config_stream.good());
  std::string config_data((std::istreambuf_iterator<char>(config_stream)),
                          (std::istreambuf_iterator<char>()));
  if (config_data.size() > 0) {
    proto_all_configs_.ParseFromString(config_data);
  }
}

ConfigReader::~ConfigReader() = default;

std::optional<ConfigReader::Config> ConfigReader::GetNextConfig() {
#ifdef WEBRTC_NETWORK_TESTER_PROTO
  if (proto_config_index_ >= proto_all_configs_.configs_size())
    return std::nullopt;
  auto proto_config = proto_all_configs_.configs(proto_config_index_++);
  RTC_DCHECK(proto_config.has_packet_send_interval_ms());
  RTC_DCHECK(proto_config.has_packet_size());
  RTC_DCHECK(proto_config.has_execution_time_ms());
  Config config;
  config.packet_send_interval_ms = proto_config.packet_send_interval_ms();
  config.packet_size = proto_config.packet_size();
  config.execution_time_ms = proto_config.execution_time_ms();
  return config;
#else
  return std::nullopt;
#endif  //  WEBRTC_NETWORK_TESTER_PROTO
}

}  // namespace webrtc
