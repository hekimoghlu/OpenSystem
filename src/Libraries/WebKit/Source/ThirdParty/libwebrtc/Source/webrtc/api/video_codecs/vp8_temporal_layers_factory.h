/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 21, 2023.
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
#ifndef API_VIDEO_CODECS_VP8_TEMPORAL_LAYERS_FACTORY_H_
#define API_VIDEO_CODECS_VP8_TEMPORAL_LAYERS_FACTORY_H_

#include <memory>

#include "api/fec_controller_override.h"
#include "api/video_codecs/video_codec.h"
#include "api/video_codecs/video_encoder.h"
#include "api/video_codecs/vp8_frame_buffer_controller.h"

namespace webrtc {

class Vp8TemporalLayersFactory : public Vp8FrameBufferControllerFactory {
 public:
  ~Vp8TemporalLayersFactory() override = default;

  std::unique_ptr<Vp8FrameBufferControllerFactory> Clone() const override;

  std::unique_ptr<Vp8FrameBufferController> Create(
      const VideoCodec& codec,
      const VideoEncoder::Settings& settings,
      FecControllerOverride* fec_controller_override) override;
};

}  // namespace webrtc

#endif  // API_VIDEO_CODECS_VP8_TEMPORAL_LAYERS_FACTORY_H_
