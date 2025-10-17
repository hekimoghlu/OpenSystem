/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 30, 2021.
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
#ifndef MODULES_AUDIO_PROCESSING_AEC3_ECHO_AUDIBILITY_H_
#define MODULES_AUDIO_PROCESSING_AEC3_ECHO_AUDIBILITY_H_

#include <stddef.h>

#include <optional>

#include "api/array_view.h"
#include "modules/audio_processing/aec3/block_buffer.h"
#include "modules/audio_processing/aec3/render_buffer.h"
#include "modules/audio_processing/aec3/spectrum_buffer.h"
#include "modules/audio_processing/aec3/stationarity_estimator.h"

namespace webrtc {

class EchoAudibility {
 public:
  explicit EchoAudibility(bool use_render_stationarity_at_init);
  ~EchoAudibility();

  EchoAudibility(const EchoAudibility&) = delete;
  EchoAudibility& operator=(const EchoAudibility&) = delete;

  // Feed new render data to the echo audibility estimator.
  void Update(const RenderBuffer& render_buffer,
              rtc::ArrayView<const float> average_reverb,
              int min_channel_delay_blocks,
              bool external_delay_seen);
  // Get the residual echo scaling.
  void GetResidualEchoScaling(bool filter_has_had_time_to_converge,
                              rtc::ArrayView<float> residual_scaling) const {
    for (size_t band = 0; band < residual_scaling.size(); ++band) {
      if (render_stationarity_.IsBandStationary(band) &&
          (filter_has_had_time_to_converge ||
           use_render_stationarity_at_init_)) {
        residual_scaling[band] = 0.f;
      } else {
        residual_scaling[band] = 1.0f;
      }
    }
  }

  // Returns true if the current render block is estimated as stationary.
  bool IsBlockStationary() const {
    return render_stationarity_.IsBlockStationary();
  }

 private:
  // Reset the EchoAudibility class.
  void Reset();

  // Updates the render stationarity flags for the current frame.
  void UpdateRenderStationarityFlags(const RenderBuffer& render_buffer,
                                     rtc::ArrayView<const float> average_reverb,
                                     int delay_blocks);

  // Updates the noise estimator with the new render data since the previous
  // call to this method.
  void UpdateRenderNoiseEstimator(const SpectrumBuffer& spectrum_buffer,
                                  const BlockBuffer& block_buffer,
                                  bool external_delay_seen);

  // Returns a bool being true if the render signal contains just close to zero
  // values.
  bool IsRenderTooLow(const BlockBuffer& block_buffer);

  std::optional<int> render_spectrum_write_prev_;
  int render_block_write_prev_;
  bool non_zero_render_seen_;
  const bool use_render_stationarity_at_init_;
  StationarityEstimator render_stationarity_;
};

}  // namespace webrtc

#endif  // MODULES_AUDIO_PROCESSING_AEC3_ECHO_AUDIBILITY_H_
