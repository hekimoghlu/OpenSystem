/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 8, 2023.
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
#include "api/video/builtin_video_bitrate_allocator_factory.h"

#include <memory>

#include "api/environment/environment.h"
#include "api/video/video_bitrate_allocator.h"
#include "api/video/video_bitrate_allocator_factory.h"
#include "api/video/video_codec_type.h"
#include "api/video_codecs/video_codec.h"
#include "modules/video_coding/svc/svc_rate_allocator.h"
#include "modules/video_coding/utility/simulcast_rate_allocator.h"

namespace webrtc {

namespace {

class BuiltinVideoBitrateAllocatorFactory
    : public VideoBitrateAllocatorFactory {
 public:
  BuiltinVideoBitrateAllocatorFactory() = default;
  ~BuiltinVideoBitrateAllocatorFactory() override = default;

  std::unique_ptr<VideoBitrateAllocator> Create(
      const Environment& env,
      const VideoCodec& codec) override {
    // TODO(https://crbug.com/webrtc/14884): Update SvcRateAllocator to
    // support simulcast and use it for VP9/AV1 simulcast as well.
    if ((codec.codecType == kVideoCodecAV1 ||
         codec.codecType == kVideoCodecVP9) &&
        codec.numberOfSimulcastStreams <= 1) {
      return std::make_unique<SvcRateAllocator>(codec, env.field_trials());
    }
    return std::make_unique<SimulcastRateAllocator>(env, codec);
  }
};

}  // namespace

std::unique_ptr<VideoBitrateAllocatorFactory>
CreateBuiltinVideoBitrateAllocatorFactory() {
  return std::make_unique<BuiltinVideoBitrateAllocatorFactory>();
}

}  // namespace webrtc
