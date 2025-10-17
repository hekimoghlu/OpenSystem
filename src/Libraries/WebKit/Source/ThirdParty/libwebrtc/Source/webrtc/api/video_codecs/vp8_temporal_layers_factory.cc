/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 10, 2024.
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
#include "api/video_codecs/vp8_temporal_layers_factory.h"

#include <algorithm>
#include <memory>
#include <utility>
#include <vector>

#include "api/fec_controller_override.h"
#include "api/video_codecs/video_codec.h"
#include "api/video_codecs/video_encoder.h"
#include "api/video_codecs/vp8_frame_buffer_controller.h"
#include "api/video_codecs/vp8_temporal_layers.h"
#include "modules/video_coding/codecs/vp8/default_temporal_layers.h"
#include "modules/video_coding/codecs/vp8/screenshare_layers.h"
#include "modules/video_coding/utility/simulcast_utility.h"
#include "rtc_base/checks.h"

namespace webrtc {

std::unique_ptr<Vp8FrameBufferController> Vp8TemporalLayersFactory::Create(
    const VideoCodec& codec,
    const VideoEncoder::Settings& /* settings */,
    FecControllerOverride* fec_controller_override) {
  std::vector<std::unique_ptr<Vp8FrameBufferController>> controllers;
  const int num_streams = SimulcastUtility::NumberOfSimulcastStreams(codec);
  RTC_DCHECK_GE(num_streams, 1);
  controllers.reserve(num_streams);

  for (int i = 0; i < num_streams; ++i) {
    int num_temporal_layers =
        SimulcastUtility::NumberOfTemporalLayers(codec, i);
    RTC_DCHECK_GE(num_temporal_layers, 1);
    if (SimulcastUtility::IsConferenceModeScreenshare(codec) && i == 0) {
      // Legacy screenshare layers supports max 2 layers.
      num_temporal_layers = std::max(2, num_temporal_layers);
      controllers.push_back(
          std::make_unique<ScreenshareLayers>(num_temporal_layers));
    } else {
      controllers.push_back(
          std::make_unique<DefaultTemporalLayers>(num_temporal_layers));
    }
  }

  return std::make_unique<Vp8TemporalLayers>(std::move(controllers),
                                             fec_controller_override);
}

std::unique_ptr<Vp8FrameBufferControllerFactory>
Vp8TemporalLayersFactory::Clone() const {
  return std::make_unique<Vp8TemporalLayersFactory>();
}

}  // namespace webrtc
