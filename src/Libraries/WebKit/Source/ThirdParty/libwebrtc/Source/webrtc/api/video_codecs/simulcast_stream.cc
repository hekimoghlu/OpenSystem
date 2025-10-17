/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 27, 2022.
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
#include "api/video_codecs/simulcast_stream.h"

#include <optional>

#include "api/video_codecs/scalability_mode.h"
#include "rtc_base/checks.h"

namespace webrtc {

unsigned char SimulcastStream::GetNumberOfTemporalLayers() const {
  return numberOfTemporalLayers;
}
void SimulcastStream::SetNumberOfTemporalLayers(unsigned char n) {
  RTC_DCHECK_GE(n, 1);
  RTC_DCHECK_LE(n, 3);
  numberOfTemporalLayers = n;
}

std::optional<ScalabilityMode> SimulcastStream::GetScalabilityMode() const {
  static const ScalabilityMode scalability_modes[3] = {
      ScalabilityMode::kL1T1,
      ScalabilityMode::kL1T2,
      ScalabilityMode::kL1T3,
  };
  if (numberOfTemporalLayers < 1 || numberOfTemporalLayers > 3) {
    return std::nullopt;
  }
  return scalability_modes[numberOfTemporalLayers - 1];
}

bool SimulcastStream::operator==(const SimulcastStream& other) const {
  return (width == other.width && height == other.height &&
          maxFramerate == other.maxFramerate &&
          numberOfTemporalLayers == other.numberOfTemporalLayers &&
          maxBitrate == other.maxBitrate &&
          targetBitrate == other.targetBitrate &&
          minBitrate == other.minBitrate && qpMax == other.qpMax &&
          active == other.active);
}

}  // namespace webrtc
