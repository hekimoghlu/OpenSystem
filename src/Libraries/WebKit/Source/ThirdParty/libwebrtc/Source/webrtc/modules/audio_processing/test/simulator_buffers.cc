/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 7, 2025.
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
#include "modules/audio_processing/test/simulator_buffers.h"

#include "modules/audio_processing/test/audio_buffer_tools.h"
#include "rtc_base/checks.h"

namespace webrtc {
namespace test {

SimulatorBuffers::SimulatorBuffers(int render_input_sample_rate_hz,
                                   int capture_input_sample_rate_hz,
                                   int render_output_sample_rate_hz,
                                   int capture_output_sample_rate_hz,
                                   size_t num_render_input_channels,
                                   size_t num_capture_input_channels,
                                   size_t num_render_output_channels,
                                   size_t num_capture_output_channels) {
  Random rand_gen(42);
  CreateConfigAndBuffer(render_input_sample_rate_hz, num_render_input_channels,
                        &rand_gen, &render_input_buffer, &render_input_config,
                        &render_input, &render_input_samples);

  CreateConfigAndBuffer(render_output_sample_rate_hz,
                        num_render_output_channels, &rand_gen,
                        &render_output_buffer, &render_output_config,
                        &render_output, &render_output_samples);

  CreateConfigAndBuffer(capture_input_sample_rate_hz,
                        num_capture_input_channels, &rand_gen,
                        &capture_input_buffer, &capture_input_config,
                        &capture_input, &capture_input_samples);

  CreateConfigAndBuffer(capture_output_sample_rate_hz,
                        num_capture_output_channels, &rand_gen,
                        &capture_output_buffer, &capture_output_config,
                        &capture_output, &capture_output_samples);

  UpdateInputBuffers();
}

SimulatorBuffers::~SimulatorBuffers() = default;

void SimulatorBuffers::CreateConfigAndBuffer(
    int sample_rate_hz,
    size_t num_channels,
    Random* rand_gen,
    std::unique_ptr<AudioBuffer>* buffer,
    StreamConfig* config,
    std::vector<float*>* buffer_data,
    std::vector<float>* buffer_data_samples) {
  int samples_per_channel = rtc::CheckedDivExact(sample_rate_hz, 100);
  *config = StreamConfig(sample_rate_hz, num_channels);
  buffer->reset(
      new AudioBuffer(config->sample_rate_hz(), config->num_channels(),
                      config->sample_rate_hz(), config->num_channels(),
                      config->sample_rate_hz(), config->num_channels()));

  buffer_data_samples->resize(samples_per_channel * num_channels);
  for (auto& v : *buffer_data_samples) {
    v = rand_gen->Rand<float>();
  }

  buffer_data->resize(num_channels);
  for (size_t ch = 0; ch < num_channels; ++ch) {
    (*buffer_data)[ch] = &(*buffer_data_samples)[ch * samples_per_channel];
  }
}

void SimulatorBuffers::UpdateInputBuffers() {
  test::CopyVectorToAudioBuffer(capture_input_config, capture_input_samples,
                                capture_input_buffer.get());
  test::CopyVectorToAudioBuffer(render_input_config, render_input_samples,
                                render_input_buffer.get());
}

}  // namespace test
}  // namespace webrtc
