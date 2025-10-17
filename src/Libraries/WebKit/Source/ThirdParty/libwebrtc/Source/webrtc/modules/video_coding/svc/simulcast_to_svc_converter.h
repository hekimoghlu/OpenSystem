/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 9, 2024.
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
#ifndef MODULES_VIDEO_CODING_SVC_SIMULCAST_TO_SVC_CONVERTER_H_
#define MODULES_VIDEO_CODING_SVC_SIMULCAST_TO_SVC_CONVERTER_H_

#include <stddef.h>

#include <memory>
#include <vector>

#include "api/video/encoded_image.h"
#include "api/video_codecs/spatial_layer.h"
#include "api/video_codecs/video_codec.h"
#include "modules/video_coding/include/video_codec_interface.h"
#include "modules/video_coding/svc/scalable_video_controller.h"
#include "rtc_base/system/rtc_export.h"

namespace webrtc {

class RTC_EXPORT SimulcastToSvcConverter {
 public:
  explicit SimulcastToSvcConverter(const VideoCodec&);
  SimulcastToSvcConverter(SimulcastToSvcConverter&&) = default;

  SimulcastToSvcConverter(const SimulcastToSvcConverter&) = delete;
  SimulcastToSvcConverter& operator=(const SimulcastToSvcConverter&) = delete;
  SimulcastToSvcConverter& operator=(SimulcastToSvcConverter&&) = default;

  ~SimulcastToSvcConverter() = default;

  static bool IsConfigSupported(const VideoCodec& codec);

  VideoCodec GetConfig() const;

  void EncodeStarted(bool force_keyframe);

  bool ConvertFrame(EncodedImage& encoded_image,
                    CodecSpecificInfo& codec_specific);

 private:
  struct LayerState {
    LayerState(ScalabilityMode scalability_mode, int num_temporal_layers);
    ~LayerState() = default;
    LayerState(const LayerState&) = delete;
    LayerState(LayerState&&) = default;

    std::unique_ptr<ScalableVideoController> video_controller;
    ScalableVideoController::LayerFrameConfig layer_config;
    bool awaiting_frame;
  };

  VideoCodec config_;

  std::vector<LayerState> layers_;
};

}  // namespace webrtc

#endif  // MODULES_VIDEO_CODING_SVC_SIMULCAST_TO_SVC_CONVERTER_H_
