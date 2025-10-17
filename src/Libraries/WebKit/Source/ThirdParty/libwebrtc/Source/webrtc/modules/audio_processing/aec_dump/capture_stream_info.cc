/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 28, 2024.
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
#include "modules/audio_processing/aec_dump/capture_stream_info.h"

namespace webrtc {

void CaptureStreamInfo::AddInput(const AudioFrameView<const float>& src) {
  auto* stream = event_->mutable_stream();

  for (int i = 0; i < src.num_channels(); ++i) {
    const auto& channel_view = src.channel(i);
    stream->add_input_channel(channel_view.begin(),
                              sizeof(float) * channel_view.size());
  }
}

void CaptureStreamInfo::AddOutput(const AudioFrameView<const float>& src) {
  auto* stream = event_->mutable_stream();

  for (int i = 0; i < src.num_channels(); ++i) {
    const auto& channel_view = src.channel(i);
    stream->add_output_channel(channel_view.begin(),
                               sizeof(float) * channel_view.size());
  }
}

void CaptureStreamInfo::AddInput(const int16_t* const data,
                                 int num_channels,
                                 int samples_per_channel) {
  auto* stream = event_->mutable_stream();
  const size_t data_size = sizeof(int16_t) * samples_per_channel * num_channels;
  stream->set_input_data(data, data_size);
}

void CaptureStreamInfo::AddOutput(const int16_t* const data,
                                  int num_channels,
                                  int samples_per_channel) {
  auto* stream = event_->mutable_stream();
  const size_t data_size = sizeof(int16_t) * samples_per_channel * num_channels;
  stream->set_output_data(data, data_size);
}

void CaptureStreamInfo::AddAudioProcessingState(
    const AecDump::AudioProcessingState& state) {
  auto* stream = event_->mutable_stream();
  stream->set_delay(state.delay);
  stream->set_drift(state.drift);
  if (state.applied_input_volume.has_value()) {
    stream->set_applied_input_volume(*state.applied_input_volume);
  }
  stream->set_keypress(state.keypress);
}
}  // namespace webrtc
