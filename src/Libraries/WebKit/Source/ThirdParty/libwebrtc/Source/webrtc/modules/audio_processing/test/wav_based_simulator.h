/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 22, 2024.
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
#ifndef MODULES_AUDIO_PROCESSING_TEST_WAV_BASED_SIMULATOR_H_
#define MODULES_AUDIO_PROCESSING_TEST_WAV_BASED_SIMULATOR_H_

#include <vector>

#include "absl/base/nullability.h"
#include "absl/strings/string_view.h"
#include "api/audio/audio_processing.h"
#include "api/scoped_refptr.h"
#include "modules/audio_processing/test/audio_processing_simulator.h"

namespace webrtc {
namespace test {

// Used to perform an audio processing simulation from wav files.
class WavBasedSimulator final : public AudioProcessingSimulator {
 public:
  WavBasedSimulator(
      const SimulationSettings& settings,
      absl::Nonnull<scoped_refptr<AudioProcessing>> audio_processing);

  WavBasedSimulator() = delete;
  WavBasedSimulator(const WavBasedSimulator&) = delete;
  WavBasedSimulator& operator=(const WavBasedSimulator&) = delete;

  ~WavBasedSimulator() override;

  // Processes the WAV input.
  void Process() override;

  // Only analyzes the data for the simulation, instead of perform any
  // processing.
  void Analyze() override;

 private:
  enum SimulationEventType {
    kProcessStream,
    kProcessReverseStream,
  };

  void Initialize();
  bool HandleProcessStreamCall();
  bool HandleProcessReverseStreamCall();
  void PrepareProcessStreamCall();
  void PrepareReverseProcessStreamCall();
  static std::vector<SimulationEventType> GetDefaultEventChain();
  static std::vector<SimulationEventType> GetCustomEventChain(
      absl::string_view filename);

  std::vector<SimulationEventType> call_chain_;
};

}  // namespace test
}  // namespace webrtc

#endif  // MODULES_AUDIO_PROCESSING_TEST_WAV_BASED_SIMULATOR_H_
