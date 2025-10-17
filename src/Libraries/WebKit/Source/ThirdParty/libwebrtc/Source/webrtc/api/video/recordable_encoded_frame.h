/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 22, 2022.
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
#ifndef API_VIDEO_RECORDABLE_ENCODED_FRAME_H_
#define API_VIDEO_RECORDABLE_ENCODED_FRAME_H_

#include <optional>

#include "api/scoped_refptr.h"
#include "api/units/timestamp.h"
#include "api/video/color_space.h"
#include "api/video/encoded_image.h"
#include "api/video/video_codec_type.h"

namespace webrtc {

// Interface for accessing recordable elements of an encoded frame.
class RecordableEncodedFrame {
 public:
  // Encoded resolution in pixels
  // TODO(bugs.webrtc.org/12114) : remove in favor of Resolution.
  struct EncodedResolution {
    bool empty() const { return width == 0 && height == 0; }

    unsigned width = 0;
    unsigned height = 0;
  };

  virtual ~RecordableEncodedFrame() = default;

  // Provides access to encoded data
  virtual rtc::scoped_refptr<const EncodedImageBufferInterface> encoded_buffer()
      const = 0;

  // Optionally returns the colorspace of the encoded frame. This can differ
  // from the eventually decoded frame's colorspace.
  virtual std::optional<webrtc::ColorSpace> color_space() const = 0;

  // Returns the codec of the encoded frame
  virtual VideoCodecType codec() const = 0;

  // Returns whether the encoded frame is a key frame
  virtual bool is_key_frame() const = 0;

  // Returns the frame's encoded resolution. May be 0x0 if the frame
  // doesn't contain resolution information
  virtual EncodedResolution resolution() const = 0;

  // Returns the computed render time
  virtual Timestamp render_time() const = 0;
};

}  // namespace webrtc

#endif  // API_VIDEO_RECORDABLE_ENCODED_FRAME_H_
