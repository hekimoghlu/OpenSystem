/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 10, 2024.
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
#ifndef MODULES_VIDEO_CODING_CODECS_TEST_ENCODED_VIDEO_FRAME_PRODUCER_H_
#define MODULES_VIDEO_CODING_CODECS_TEST_ENCODED_VIDEO_FRAME_PRODUCER_H_

#include <stdint.h>

#include <vector>

#include "api/transport/rtp/dependency_descriptor.h"
#include "api/video/encoded_image.h"
#include "api/video_codecs/video_encoder.h"
#include "modules/video_coding/include/video_codec_interface.h"

namespace webrtc {

// Wrapper around VideoEncoder::Encode for convenient input (generates frames)
// and output (returns encoded frames instead of passing them to callback)
class EncodedVideoFrameProducer {
 public:
  struct EncodedFrame {
    EncodedImage encoded_image;
    CodecSpecificInfo codec_specific_info;
  };

  // `encoder` should be initialized, but shouldn't have `EncoderCallback` set.
  explicit EncodedVideoFrameProducer(VideoEncoder& encoder)
      : encoder_(encoder) {}
  EncodedVideoFrameProducer(const EncodedVideoFrameProducer&) = delete;
  EncodedVideoFrameProducer& operator=(const EncodedVideoFrameProducer&) =
      delete;

  // Number of the input frames to pass to the encoder.
  EncodedVideoFrameProducer& SetNumInputFrames(int value);
  // Encode next frame as key frame.
  EncodedVideoFrameProducer& ForceKeyFrame();
  // Resolution of the input frames.
  EncodedVideoFrameProducer& SetResolution(RenderResolution value);

  EncodedVideoFrameProducer& SetFramerateFps(int value);

  EncodedVideoFrameProducer& SetRtpTimestamp(uint32_t value);

  EncodedVideoFrameProducer& SetPresentationTimestamp(Timestamp value);

  // Generates input video frames and encodes them with `encoder` provided
  // in the constructor. Returns frame passed to the `OnEncodedImage` by
  // wraping `EncodedImageCallback` underneath.
  std::vector<EncodedFrame> Encode();

 private:
  VideoEncoder& encoder_;

  uint32_t rtp_timestamp_ = 1000;
  Timestamp presentation_timestamp_ = Timestamp::Micros(1000);
  int num_input_frames_ = 1;
  int framerate_fps_ = 30;
  RenderResolution resolution_ = {320, 180};
  std::vector<VideoFrameType> next_frame_type_ = {
      VideoFrameType::kVideoFrameKey};
};

inline EncodedVideoFrameProducer& EncodedVideoFrameProducer::SetNumInputFrames(
    int value) {
  RTC_DCHECK_GT(value, 0);
  num_input_frames_ = value;
  return *this;
}

inline EncodedVideoFrameProducer& EncodedVideoFrameProducer::ForceKeyFrame() {
  next_frame_type_ = {VideoFrameType::kVideoFrameKey};
  return *this;
}

inline EncodedVideoFrameProducer& EncodedVideoFrameProducer::SetResolution(
    RenderResolution value) {
  resolution_ = value;
  return *this;
}

inline EncodedVideoFrameProducer& EncodedVideoFrameProducer::SetFramerateFps(
    int value) {
  RTC_DCHECK_GT(value, 0);
  framerate_fps_ = value;
  return *this;
}

inline EncodedVideoFrameProducer& EncodedVideoFrameProducer::SetRtpTimestamp(
    uint32_t value) {
  rtp_timestamp_ = value;
  return *this;
}

inline EncodedVideoFrameProducer&
EncodedVideoFrameProducer::SetPresentationTimestamp(Timestamp value) {
  presentation_timestamp_ = value;
  return *this;
}
}  // namespace webrtc
#endif  // MODULES_VIDEO_CODING_CODECS_TEST_ENCODED_VIDEO_FRAME_PRODUCER_H_
