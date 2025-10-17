/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 18, 2024.
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
#include "call/adaptation/video_stream_input_state.h"

#include "api/video_codecs/video_encoder.h"

namespace webrtc {

VideoStreamInputState::VideoStreamInputState()
    : has_input_(false),
      frame_size_pixels_(std::nullopt),
      frames_per_second_(0),
      video_codec_type_(VideoCodecType::kVideoCodecGeneric),
      min_pixels_per_frame_(kDefaultMinPixelsPerFrame),
      single_active_stream_pixels_(std::nullopt) {}

void VideoStreamInputState::set_has_input(bool has_input) {
  has_input_ = has_input;
}

void VideoStreamInputState::set_frame_size_pixels(
    std::optional<int> frame_size_pixels) {
  frame_size_pixels_ = frame_size_pixels;
}

void VideoStreamInputState::set_frames_per_second(int frames_per_second) {
  frames_per_second_ = frames_per_second;
}

void VideoStreamInputState::set_video_codec_type(
    VideoCodecType video_codec_type) {
  video_codec_type_ = video_codec_type;
}

void VideoStreamInputState::set_min_pixels_per_frame(int min_pixels_per_frame) {
  min_pixels_per_frame_ = min_pixels_per_frame;
}

void VideoStreamInputState::set_single_active_stream_pixels(
    std::optional<int> single_active_stream_pixels) {
  single_active_stream_pixels_ = single_active_stream_pixels;
}

bool VideoStreamInputState::has_input() const {
  return has_input_;
}

std::optional<int> VideoStreamInputState::frame_size_pixels() const {
  return frame_size_pixels_;
}

int VideoStreamInputState::frames_per_second() const {
  return frames_per_second_;
}

VideoCodecType VideoStreamInputState::video_codec_type() const {
  return video_codec_type_;
}

int VideoStreamInputState::min_pixels_per_frame() const {
  return min_pixels_per_frame_;
}

std::optional<int> VideoStreamInputState::single_active_stream_pixels() const {
  return single_active_stream_pixels_;
}

bool VideoStreamInputState::HasInputFrameSizeAndFramesPerSecond() const {
  return has_input_ && frame_size_pixels_.has_value();
}

}  // namespace webrtc
