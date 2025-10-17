/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 3, 2021.
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
#ifndef MODULES_AUDIO_PROCESSING_TEST_AEC_DUMP_BASED_SIMULATOR_H_
#define MODULES_AUDIO_PROCESSING_TEST_AEC_DUMP_BASED_SIMULATOR_H_

#include <cstdio>
#include <fstream>
#include <memory>

#include "absl/base/nullability.h"
#include "api/audio/audio_processing.h"
#include "api/scoped_refptr.h"
#include "common_audio/channel_buffer.h"
#include "modules/audio_processing/test/audio_processing_simulator.h"
#include "modules/audio_processing/test/test_utils.h"

#ifdef WEBRTC_ANDROID_PLATFORM_BUILD
#include "external/webrtc/webrtc/modules/audio_processing/debug.pb.h"
#else
#include "modules/audio_processing/debug.pb.h"
#endif

namespace webrtc {
namespace test {

// Used to perform an audio processing simulation from an aec dump.
class AecDumpBasedSimulator final : public AudioProcessingSimulator {
 public:
  AecDumpBasedSimulator(
      const SimulationSettings& settings,
      absl::Nonnull<scoped_refptr<AudioProcessing>> audio_processing);

  AecDumpBasedSimulator() = delete;
  AecDumpBasedSimulator(const AecDumpBasedSimulator&) = delete;
  AecDumpBasedSimulator& operator=(const AecDumpBasedSimulator&) = delete;

  ~AecDumpBasedSimulator() override;

  // Processes the messages in the aecdump file.
  void Process() override;

  // Analyzes the data in the aecdump file and reports the resulting statistics.
  void Analyze() override;

 private:
  void HandleEvent(const webrtc::audioproc::Event& event_msg,
                   int& num_forward_chunks_processed,
                   int& init_index);
  void HandleMessage(const webrtc::audioproc::Init& msg, int init_index);
  void HandleMessage(const webrtc::audioproc::Stream& msg);
  void HandleMessage(const webrtc::audioproc::ReverseStream& msg);
  void HandleMessage(const webrtc::audioproc::Config& msg);
  void HandleMessage(const webrtc::audioproc::RuntimeSetting& msg);
  void PrepareProcessStreamCall(const webrtc::audioproc::Stream& msg);
  void PrepareReverseProcessStreamCall(
      const webrtc::audioproc::ReverseStream& msg);
  void VerifyProcessStreamBitExactness(const webrtc::audioproc::Stream& msg);
  void MaybeOpenCallOrderFile();
  enum InterfaceType {
    kFixedInterface,
    kFloatInterface,
    kNotSpecified,
  };

  FILE* dump_input_file_;
  std::unique_ptr<ChannelBuffer<float>> artificial_nearend_buf_;
  std::unique_ptr<ChannelBufferWavReader> artificial_nearend_buffer_reader_;
  bool artificial_nearend_eof_reported_ = false;
  InterfaceType interface_used_ = InterfaceType::kNotSpecified;
  std::unique_ptr<std::ofstream> call_order_output_file_;
  bool finished_processing_specified_init_block_ = false;
};

}  // namespace test
}  // namespace webrtc

#endif  // MODULES_AUDIO_PROCESSING_TEST_AEC_DUMP_BASED_SIMULATOR_H_
