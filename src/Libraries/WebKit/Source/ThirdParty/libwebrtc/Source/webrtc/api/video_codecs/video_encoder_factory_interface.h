/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 19, 2023.
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
#ifndef API_VIDEO_CODECS_VIDEO_ENCODER_FACTORY_INTERFACE_H_
#define API_VIDEO_CODECS_VIDEO_ENCODER_FACTORY_INTERFACE_H_

#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/types/variant.h"
#include "api/units/time_delta.h"
#include "api/video/resolution.h"
#include "api/video/video_frame_buffer.h"
#include "api/video_codecs/video_encoder_interface.h"
#include "api/video_codecs/video_encoding_general.h"
#include "rtc_base/numerics/rational.h"

namespace webrtc {
using FrameType = VideoEncoderInterface::FrameType;

// NOTE: This class is still under development and may change without notice.
class VideoEncoderFactoryInterface {
 public:
  enum class RateControlMode { kCqp, kCbr };

  struct Capabilities {
    struct PredictionConstraints {
      enum class BufferSpaceType {
        kMultiInstance,  // multiple independent sets of buffers
        kMultiKeyframe,  // single set of buffers, but can store multiple
                         // keyframes simultaneously.
        kSingleKeyframe  // single set of buffers, can only store one keyframe
                         // at a time.
      };

      int num_buffers;
      int max_references;
      int max_temporal_layers;

      BufferSpaceType buffer_space_type;
      int max_spatial_layers;
      std::vector<Rational> scaling_factors;

      std::vector<FrameType> supported_frame_types;
    } prediction_constraints;

    struct InputConstraints {
      Resolution min;
      Resolution max;
      int pixel_alignment;
      std::vector<VideoFrameBuffer::Type> input_formats;
    } input_constraints;

    std::vector<EncodingFormat> encoding_formats;

    struct BitrateControl {
      std::pair<int, int> qp_range;
      std::vector<RateControlMode> rc_modes;
    } rate_control;

    struct Performance {
      bool encode_on_calling_thread;
      std::pair<int, int> min_max_effort_level;
    } performance;
  };

  struct StaticEncoderSettings {
    struct Cqp {};
    struct Cbr {
      // TD: Should there be an intial buffer size?
      TimeDelta max_buffer_size;
      TimeDelta target_buffer_size;
    };

    Resolution max_encode_dimensions;
    EncodingFormat encoding_format;
    absl::variant<Cqp, Cbr> rc_mode;
    int max_number_of_threads;
  };

  virtual ~VideoEncoderFactoryInterface() = default;

  virtual std::string CodecName() const = 0;
  virtual std::string ImplementationName() const = 0;
  virtual std::map<std::string, std::string> CodecSpecifics() const = 0;

  virtual Capabilities GetEncoderCapabilities() const = 0;
  virtual std::unique_ptr<VideoEncoderInterface> CreateEncoder(
      const StaticEncoderSettings& settings,
      const std::map<std::string, std::string>& encoder_specific_settings) = 0;
};

}  // namespace webrtc
#endif  // API_VIDEO_CODECS_VIDEO_ENCODER_FACTORY_INTERFACE_H_
