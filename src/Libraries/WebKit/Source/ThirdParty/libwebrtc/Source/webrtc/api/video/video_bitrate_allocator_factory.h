/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 10, 2025.
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
#ifndef API_VIDEO_VIDEO_BITRATE_ALLOCATOR_FACTORY_H_
#define API_VIDEO_VIDEO_BITRATE_ALLOCATOR_FACTORY_H_

#include <memory>

#include "api/environment/environment.h"
#include "api/video/video_bitrate_allocator.h"
#include "api/video_codecs/video_codec.h"

namespace webrtc {

// A factory that creates VideoBitrateAllocator.
// NOTE: This class is still under development and may change without notice.
class VideoBitrateAllocatorFactory {
 public:
  virtual ~VideoBitrateAllocatorFactory() = default;

  // Creates a VideoBitrateAllocator for a specific video codec.
  virtual std::unique_ptr<VideoBitrateAllocator> Create(
      const Environment& env,
      const VideoCodec& codec) = 0;
};

}  // namespace webrtc

#endif  // API_VIDEO_VIDEO_BITRATE_ALLOCATOR_FACTORY_H_
