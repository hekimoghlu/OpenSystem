/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 9, 2022.
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
#ifndef MODULES_AUDIO_PROCESSING_TEST_SIMULATOR_BUFFERS_H_
#define MODULES_AUDIO_PROCESSING_TEST_SIMULATOR_BUFFERS_H_

#include <memory>
#include <vector>

#include "api/audio/audio_processing.h"
#include "modules/audio_processing/audio_buffer.h"
#include "rtc_base/random.h"

namespace webrtc {
namespace test {

struct SimulatorBuffers {
  SimulatorBuffers(int render_input_sample_rate_hz,
                   int capture_input_sample_rate_hz,
                   int render_output_sample_rate_hz,
                   int capture_output_sample_rate_hz,
                   size_t num_render_input_channels,
                   size_t num_capture_input_channels,
                   size_t num_render_output_channels,
                   size_t num_capture_output_channels);
  ~SimulatorBuffers();

  void CreateConfigAndBuffer(int sample_rate_hz,
                             size_t num_channels,
                             Random* rand_gen,
                             std::unique_ptr<AudioBuffer>* buffer,
                             StreamConfig* config,
                             std::vector<float*>* buffer_data,
                             std::vector<float>* buffer_data_samples);

  void UpdateInputBuffers();

  std::unique_ptr<AudioBuffer> render_input_buffer;
  std::unique_ptr<AudioBuffer> capture_input_buffer;
  std::unique_ptr<AudioBuffer> render_output_buffer;
  std::unique_ptr<AudioBuffer> capture_output_buffer;
  StreamConfig render_input_config;
  StreamConfig capture_input_config;
  StreamConfig render_output_config;
  StreamConfig capture_output_config;
  std::vector<float*> render_input;
  std::vector<float> render_input_samples;
  std::vector<float*> capture_input;
  std::vector<float> capture_input_samples;
  std::vector<float*> render_output;
  std::vector<float> render_output_samples;
  std::vector<float*> capture_output;
  std::vector<float> capture_output_samples;
};

}  // namespace test
}  // namespace webrtc

#endif  // MODULES_AUDIO_PROCESSING_TEST_SIMULATOR_BUFFERS_H_
