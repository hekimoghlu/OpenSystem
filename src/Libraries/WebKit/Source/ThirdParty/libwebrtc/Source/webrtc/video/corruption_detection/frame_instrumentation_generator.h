/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 14, 2024.
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
#ifndef VIDEO_CORRUPTION_DETECTION_FRAME_INSTRUMENTATION_GENERATOR_H_
#define VIDEO_CORRUPTION_DETECTION_FRAME_INSTRUMENTATION_GENERATOR_H_

#include <cstdint>
#include <map>
#include <optional>
#include <queue>

#include "absl/types/variant.h"
#include "api/video/encoded_image.h"
#include "api/video/video_codec_type.h"
#include "api/video/video_frame.h"
#include "common_video/frame_instrumentation_data.h"
#include "video/corruption_detection/halton_frame_sampler.h"

namespace webrtc {

class FrameInstrumentationGenerator {
 public:
  FrameInstrumentationGenerator() = delete;
  explicit FrameInstrumentationGenerator(VideoCodecType video_codec_type);

  FrameInstrumentationGenerator(const FrameInstrumentationGenerator&) = delete;
  FrameInstrumentationGenerator& operator=(
      const FrameInstrumentationGenerator&) = delete;

  ~FrameInstrumentationGenerator() = default;

  void OnCapturedFrame(VideoFrame frame);
  std::optional<
      absl::variant<FrameInstrumentationSyncData, FrameInstrumentationData>>
  OnEncodedImage(const EncodedImage& encoded_image);

  // Returns `std::nullopt` if there is no context for the given layer.
  std::optional<int> GetHaltonSequenceIndex(int layer_id) const;
  void SetHaltonSequenceIndex(int index, int layer_id);

  int GetLayerId(const EncodedImage& encoded_image) const;

 private:
  struct Context {
    HaltonFrameSampler frame_sampler;
    uint32_t rtp_timestamp_of_last_key_frame = 0;
  };

  // Incoming video frames in capture order.
  std::queue<VideoFrame> captured_frames_;
  // Map from spatial or simulcast index to sampling context.
  std::map<int, Context> contexts_;
  const VideoCodecType video_codec_type_;
};

}  // namespace webrtc

#endif  // VIDEO_CORRUPTION_DETECTION_FRAME_INSTRUMENTATION_GENERATOR_H_
