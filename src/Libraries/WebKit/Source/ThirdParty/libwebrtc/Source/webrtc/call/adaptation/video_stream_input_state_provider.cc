/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 30, 2022.
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
#include "call/adaptation/video_stream_input_state_provider.h"

#include "call/adaptation/video_stream_adapter.h"

namespace webrtc {

VideoStreamInputStateProvider::VideoStreamInputStateProvider(
    VideoStreamEncoderObserver* frame_rate_provider)
    : frame_rate_provider_(frame_rate_provider) {}

VideoStreamInputStateProvider::~VideoStreamInputStateProvider() {}

void VideoStreamInputStateProvider::OnHasInputChanged(bool has_input) {
  MutexLock lock(&mutex_);
  input_state_.set_has_input(has_input);
}

void VideoStreamInputStateProvider::OnFrameSizeObserved(int frame_size_pixels) {
  RTC_DCHECK_GT(frame_size_pixels, 0);
  MutexLock lock(&mutex_);
  input_state_.set_frame_size_pixels(frame_size_pixels);
}

void VideoStreamInputStateProvider::OnEncoderSettingsChanged(
    EncoderSettings encoder_settings) {
  MutexLock lock(&mutex_);
  input_state_.set_video_codec_type(
      encoder_settings.encoder_config().codec_type);
  input_state_.set_min_pixels_per_frame(
      encoder_settings.encoder_info().scaling_settings.min_pixels_per_frame);
  input_state_.set_single_active_stream_pixels(
      VideoStreamAdapter::GetSingleActiveLayerPixels(
          encoder_settings.video_codec()));
}

VideoStreamInputState VideoStreamInputStateProvider::InputState() {
  // GetInputFrameRate() is thread-safe.
  int input_fps = frame_rate_provider_->GetInputFrameRate();
  MutexLock lock(&mutex_);
  input_state_.set_frames_per_second(input_fps);
  return input_state_;
}

}  // namespace webrtc
