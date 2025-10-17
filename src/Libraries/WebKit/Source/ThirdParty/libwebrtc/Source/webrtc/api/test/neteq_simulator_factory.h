/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 2, 2021.
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
#ifndef API_TEST_NETEQ_SIMULATOR_FACTORY_H_
#define API_TEST_NETEQ_SIMULATOR_FACTORY_H_

#include <cstdint>
#include <memory>
#include <optional>
#include <string>

#include "absl/strings/string_view.h"
#include "api/neteq/neteq_factory.h"
#include "api/test/neteq_simulator.h"

namespace webrtc {
namespace test {

class NetEqTestFactory;

class NetEqSimulatorFactory {
 public:
  NetEqSimulatorFactory();
  ~NetEqSimulatorFactory();
  struct Config {
    // The maximum allowed number of packets in the jitter buffer.
    int max_nr_packets_in_buffer = 0;
    // The number of audio packets to insert at the start of the simulation.
    // Since the simulation is done with a replacement audio file, these
    // artificial packets will take a small piece of that replacement audio.
    int initial_dummy_packets = 0;
    // The number of simulation steps to skip at the start of the simulation.
    // This removes incoming packets and GetAudio events from the start of the
    // simulation, until the requested number of GetAudio events has been
    // removed.
    int skip_get_audio_events = 0;
    // A WebRTC field trial string to be used during the simulation.
    std::string field_trial_string;
    // A filename for the generated output audio file.
    std::optional<std::string> output_audio_filename;
    // A filename for the python plot.
    std::optional<std::string> python_plot_filename;
    // A filename for the text log.
    std::optional<std::string> text_log_filename;
    // A custom NetEqFactory can be used.
    NetEqFactory* neteq_factory = nullptr;
    // The SSRC to use for the simulation.
    std::optional<uint32_t> ssrc_filter;
  };
  std::unique_ptr<NetEqSimulator> CreateSimulatorFromFile(
      absl::string_view event_log_filename,
      absl::string_view replacement_audio_filename,
      Config simulation_config);
  // The same as above, but pass the file contents as a string.
  std::unique_ptr<NetEqSimulator> CreateSimulatorFromString(
      absl::string_view event_log_file_contents,
      absl::string_view replacement_audio_file,
      Config simulation_config);

 private:
  std::unique_ptr<NetEqTestFactory> factory_;
};

}  // namespace test
}  // namespace webrtc

#endif  // API_TEST_NETEQ_SIMULATOR_FACTORY_H_
