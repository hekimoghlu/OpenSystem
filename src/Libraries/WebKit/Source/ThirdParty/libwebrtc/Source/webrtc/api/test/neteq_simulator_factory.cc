/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 11, 2023.
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
#include "api/test/neteq_simulator_factory.h"

#include <memory>
#include <string>

#include "absl/strings/string_view.h"
#include "api/test/neteq_simulator.h"
#include "modules/audio_coding/neteq/tools/neteq_test_factory.h"

namespace webrtc {
namespace test {
namespace {
NetEqTestFactory::Config convertConfig(
    const NetEqSimulatorFactory::Config& simulation_config,
    absl::string_view replacement_audio_filename) {
  NetEqTestFactory::Config config;
  config.replacement_audio_file = std::string(replacement_audio_filename);
  config.max_nr_packets_in_buffer = simulation_config.max_nr_packets_in_buffer;
  config.initial_dummy_packets = simulation_config.initial_dummy_packets;
  config.skip_get_audio_events = simulation_config.skip_get_audio_events;
  config.field_trial_string = simulation_config.field_trial_string;
  config.output_audio_filename = simulation_config.output_audio_filename;
  config.pythonplot = simulation_config.python_plot_filename.has_value();
  config.plot_scripts_basename = simulation_config.python_plot_filename;
  config.textlog = simulation_config.text_log_filename.has_value();
  config.textlog_filename = simulation_config.text_log_filename;
  config.ssrc_filter = simulation_config.ssrc_filter;
  return config;
}
}  // namespace

NetEqSimulatorFactory::NetEqSimulatorFactory()
    : factory_(std::make_unique<NetEqTestFactory>()) {}

NetEqSimulatorFactory::~NetEqSimulatorFactory() = default;

std::unique_ptr<NetEqSimulator> NetEqSimulatorFactory::CreateSimulatorFromFile(
    absl::string_view event_log_filename,
    absl::string_view replacement_audio_filename,
    Config simulation_config) {
  NetEqTestFactory::Config config =
      convertConfig(simulation_config, replacement_audio_filename);
  return factory_->InitializeTestFromFile(
      std::string(event_log_filename), simulation_config.neteq_factory, config);
}

std::unique_ptr<NetEqSimulator>
NetEqSimulatorFactory::CreateSimulatorFromString(
    absl::string_view event_log_file_contents,
    absl::string_view replacement_audio_filename,
    Config simulation_config) {
  NetEqTestFactory::Config config =
      convertConfig(simulation_config, replacement_audio_filename);
  return factory_->InitializeTestFromString(
      std::string(event_log_file_contents), simulation_config.neteq_factory,
      config);
}

}  // namespace test
}  // namespace webrtc
