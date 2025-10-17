/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 30, 2022.
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
#ifndef API_VIDEO_CODECS_VIDEO_ENCODER_INTERFACE_H_
#define API_VIDEO_CODECS_VIDEO_ENCODER_INTERFACE_H_

#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include "absl/types/variant.h"
#include "api/array_view.h"
#include "api/scoped_refptr.h"
#include "api/units/data_rate.h"
#include "api/units/data_size.h"
#include "api/units/time_delta.h"
#include "api/units/timestamp.h"
#include "api/video/resolution.h"
#include "api/video/video_frame_buffer.h"
#include "api/video_codecs/video_codec.h"

namespace webrtc {
// NOTE: This class is still under development and may change without notice.
class VideoEncoderInterface {
 public:
  virtual ~VideoEncoderInterface() = default;
  enum class FrameType { kKeyframe, kStartFrame, kDeltaFrame };

  struct EncodingError {};
  struct EncodedData {
    FrameType frame_type;
    int encoded_qp;
  };
  using EncodeResult = absl::variant<EncodingError, EncodedData>;

  struct FrameOutput {
    virtual ~FrameOutput() = default;
    virtual rtc::ArrayView<uint8_t> GetBitstreamOutputBuffer(DataSize size) = 0;
    virtual void EncodeComplete(const EncodeResult& encode_result) = 0;
  };

  struct TemporalUnitSettings {
    VideoCodecMode content_hint = VideoCodecMode::kRealtimeVideo;
    Timestamp presentation_timestamp;
  };

  struct FrameEncodeSettings {
    struct Cbr {
      TimeDelta duration;
      DataRate target_bitrate;
    };

    struct Cqp {
      int target_qp;
    };

    absl::variant<Cqp, Cbr> rate_options;

    FrameType frame_type = FrameType::kDeltaFrame;
    int temporal_id = 0;
    int spatial_id = 0;
    Resolution resolution;
    std::vector<int> reference_buffers;
    std::optional<int> update_buffer;
    int effort_level = 0;

    std::unique_ptr<FrameOutput> frame_output;
  };

  virtual void Encode(rtc::scoped_refptr<webrtc::VideoFrameBuffer> frame_buffer,
                      const TemporalUnitSettings& settings,
                      std::vector<FrameEncodeSettings> frame_settings) = 0;
};

}  // namespace webrtc
#endif  // API_VIDEO_CODECS_VIDEO_ENCODER_INTERFACE_H_
