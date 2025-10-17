/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 15, 2025.
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
#include "modules/audio_processing/test/wav_based_simulator.h"

#include <stdio.h>

#include <iostream>
#include <memory>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/strings/string_view.h"
#include "api/audio/audio_processing.h"
#include "api/scoped_refptr.h"
#include "common_audio/wav_file.h"
#include "modules/audio_processing/test/audio_processing_simulator.h"
#include "modules/audio_processing/test/test_utils.h"
#include "rtc_base/checks.h"
#include "rtc_base/system/file_wrapper.h"

namespace webrtc {
namespace test {

std::vector<WavBasedSimulator::SimulationEventType>
WavBasedSimulator::GetCustomEventChain(absl::string_view filename) {
  std::vector<WavBasedSimulator::SimulationEventType> call_chain;
  FileWrapper file_wrapper = FileWrapper::OpenReadOnly(filename);

  RTC_CHECK(file_wrapper.is_open())
      << "Could not open the custom call order file, reverting "
         "to using the default call order";

  char c;
  size_t num_read = file_wrapper.Read(&c, sizeof(char));
  while (num_read > 0) {
    switch (c) {
      case 'r':
        call_chain.push_back(SimulationEventType::kProcessReverseStream);
        break;
      case 'c':
        call_chain.push_back(SimulationEventType::kProcessStream);
        break;
      case '\n':
        break;
      default:
        RTC_FATAL() << "Incorrect custom call order file";
    }

    num_read = file_wrapper.Read(&c, sizeof(char));
  }

  return call_chain;
}

WavBasedSimulator::WavBasedSimulator(
    const SimulationSettings& settings,
    absl::Nonnull<scoped_refptr<AudioProcessing>> audio_processing)
    : AudioProcessingSimulator(settings, std::move(audio_processing)) {
  if (settings_.call_order_input_filename) {
    call_chain_ = WavBasedSimulator::GetCustomEventChain(
        *settings_.call_order_input_filename);
  } else {
    call_chain_ = WavBasedSimulator::GetDefaultEventChain();
  }
}

WavBasedSimulator::~WavBasedSimulator() = default;

std::vector<WavBasedSimulator::SimulationEventType>
WavBasedSimulator::GetDefaultEventChain() {
  std::vector<WavBasedSimulator::SimulationEventType> call_chain(2);
  call_chain[0] = SimulationEventType::kProcessStream;
  call_chain[1] = SimulationEventType::kProcessReverseStream;
  return call_chain;
}

void WavBasedSimulator::PrepareProcessStreamCall() {
  if (settings_.fixed_interface) {
    fwd_frame_.CopyFrom(*in_buf_);
  }
  ap_->set_stream_key_pressed(settings_.override_key_pressed.value_or(false));

  if (!settings_.use_stream_delay || *settings_.use_stream_delay) {
    RTC_CHECK_EQ(AudioProcessing::kNoError,
                 ap_->set_stream_delay_ms(
                     settings_.stream_delay ? *settings_.stream_delay : 0));
  }
}

void WavBasedSimulator::PrepareReverseProcessStreamCall() {
  if (settings_.fixed_interface) {
    rev_frame_.CopyFrom(*reverse_in_buf_);
  }
}

void WavBasedSimulator::Process() {
  ConfigureAudioProcessor();

  Initialize();

  bool samples_left_to_process = true;
  int call_chain_index = 0;
  int capture_frames_since_init = 0;
  constexpr int kInitIndex = 1;
  while (samples_left_to_process) {
    switch (call_chain_[call_chain_index]) {
      case SimulationEventType::kProcessStream:
        SelectivelyToggleDataDumping(kInitIndex, capture_frames_since_init);

        samples_left_to_process = HandleProcessStreamCall();
        ++capture_frames_since_init;
        break;
      case SimulationEventType::kProcessReverseStream:
        if (settings_.reverse_input_filename) {
          samples_left_to_process = HandleProcessReverseStreamCall();
        }
        break;
      default:
        RTC_CHECK_NOTREACHED();
    }

    call_chain_index = (call_chain_index + 1) % call_chain_.size();
  }

  DetachAecDump();
}

void WavBasedSimulator::Analyze() {
  std::cout << "Inits:" << std::endl;
  std::cout << "1: -->" << std::endl;
  std::cout << " Time:" << std::endl;
  std::cout << "  Capture: 0 s (0 frames) " << std::endl;
  std::cout << "  Render: 0 s (0 frames)" << std::endl;
}

bool WavBasedSimulator::HandleProcessStreamCall() {
  bool samples_left_to_process = buffer_reader_->Read(in_buf_.get());
  if (samples_left_to_process) {
    PrepareProcessStreamCall();
    ProcessStream(settings_.fixed_interface);
  }
  return samples_left_to_process;
}

bool WavBasedSimulator::HandleProcessReverseStreamCall() {
  bool samples_left_to_process =
      reverse_buffer_reader_->Read(reverse_in_buf_.get());
  if (samples_left_to_process) {
    PrepareReverseProcessStreamCall();
    ProcessReverseStream(settings_.fixed_interface);
  }
  return samples_left_to_process;
}

void WavBasedSimulator::Initialize() {
  std::unique_ptr<WavReader> in_file(
      new WavReader(settings_.input_filename->c_str()));
  int input_sample_rate_hz = in_file->sample_rate();
  int input_num_channels = in_file->num_channels();
  buffer_reader_.reset(new ChannelBufferWavReader(std::move(in_file)));

  int output_sample_rate_hz = settings_.output_sample_rate_hz
                                  ? *settings_.output_sample_rate_hz
                                  : input_sample_rate_hz;
  int output_num_channels = settings_.output_num_channels
                                ? *settings_.output_num_channels
                                : input_num_channels;

  int reverse_sample_rate_hz = 48000;
  int reverse_num_channels = 1;
  int reverse_output_sample_rate_hz = 48000;
  int reverse_output_num_channels = 1;
  if (settings_.reverse_input_filename) {
    std::unique_ptr<WavReader> reverse_in_file(
        new WavReader(settings_.reverse_input_filename->c_str()));
    reverse_sample_rate_hz = reverse_in_file->sample_rate();
    reverse_num_channels = reverse_in_file->num_channels();
    reverse_buffer_reader_.reset(
        new ChannelBufferWavReader(std::move(reverse_in_file)));

    reverse_output_sample_rate_hz =
        settings_.reverse_output_sample_rate_hz
            ? *settings_.reverse_output_sample_rate_hz
            : reverse_sample_rate_hz;
    reverse_output_num_channels = settings_.reverse_output_num_channels
                                      ? *settings_.reverse_output_num_channels
                                      : reverse_num_channels;
  }

  SetupBuffersConfigsOutputs(
      input_sample_rate_hz, output_sample_rate_hz, reverse_sample_rate_hz,
      reverse_output_sample_rate_hz, input_num_channels, output_num_channels,
      reverse_num_channels, reverse_output_num_channels);
}

}  // namespace test
}  // namespace webrtc
