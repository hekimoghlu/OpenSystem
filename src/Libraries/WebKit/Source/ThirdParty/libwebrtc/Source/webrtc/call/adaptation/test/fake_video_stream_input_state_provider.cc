/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 14, 2024.
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
#include "call/adaptation/test/fake_video_stream_input_state_provider.h"

namespace webrtc {

FakeVideoStreamInputStateProvider::FakeVideoStreamInputStateProvider()
    : VideoStreamInputStateProvider(nullptr) {}

FakeVideoStreamInputStateProvider::~FakeVideoStreamInputStateProvider() =
    default;

void FakeVideoStreamInputStateProvider::SetInputState(
    int input_pixels,
    int input_fps,
    int min_pixels_per_frame) {
  fake_input_state_.set_has_input(true);
  fake_input_state_.set_frame_size_pixels(input_pixels);
  fake_input_state_.set_frames_per_second(input_fps);
  fake_input_state_.set_min_pixels_per_frame(min_pixels_per_frame);
}

VideoStreamInputState FakeVideoStreamInputStateProvider::InputState() {
  return fake_input_state_;
}

}  // namespace webrtc
