/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 21, 2025.
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
#ifndef API_VIDEO_CODECS_SIMULCAST_STREAM_H_
#define API_VIDEO_CODECS_SIMULCAST_STREAM_H_

#include <optional>

#include "api/video_codecs/scalability_mode.h"
#include "rtc_base/system/rtc_export.h"

namespace webrtc {

// TODO(bugs.webrtc.org/6883): Unify with struct VideoStream, part of
// VideoEncoderConfig.
struct RTC_EXPORT SimulcastStream {
  // Temporary utility methods for transition from numberOfTemporalLayers
  // setting to ScalabilityMode.
  unsigned char GetNumberOfTemporalLayers() const;
  std::optional<ScalabilityMode> GetScalabilityMode() const;
  void SetNumberOfTemporalLayers(unsigned char n);

  bool operator==(const SimulcastStream& other) const;
  bool operator!=(const SimulcastStream& other) const {
    return !(*this == other);
  }

  int width = 0;
  int height = 0;
  float maxFramerate = 0;  // fps.
  unsigned char numberOfTemporalLayers = 1;
  unsigned int maxBitrate = 0;     // kilobits/sec.
  unsigned int targetBitrate = 0;  // kilobits/sec.
  unsigned int minBitrate = 0;     // kilobits/sec.
  unsigned int qpMax = 0;          // minimum quality
  bool active = false;             // encoded and sent.
};

}  // namespace webrtc
#endif  // API_VIDEO_CODECS_SIMULCAST_STREAM_H_
