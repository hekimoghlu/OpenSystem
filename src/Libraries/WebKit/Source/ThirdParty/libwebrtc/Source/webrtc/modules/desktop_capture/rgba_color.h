/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 20, 2023.
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
#ifndef MODULES_DESKTOP_CAPTURE_RGBA_COLOR_H_
#define MODULES_DESKTOP_CAPTURE_RGBA_COLOR_H_

#include <stdint.h>

#include "modules/desktop_capture/desktop_frame.h"

namespace webrtc {

// A four-byte structure to store a color in BGRA format. This structure also
// provides functions to be created from uint8_t array, say,
// DesktopFrame::data(). It always uses BGRA order for internal storage to match
// DesktopFrame::data().
struct RgbaColor final {
  // Creates a color with BGRA channels.
  RgbaColor(uint8_t blue, uint8_t green, uint8_t red, uint8_t alpha);

  // Creates a color with BGR channels, and set alpha channel to 255 (opaque).
  RgbaColor(uint8_t blue, uint8_t green, uint8_t red);

  // Creates a color from four-byte in BGRA order, i.e. DesktopFrame::data().
  explicit RgbaColor(const uint8_t* bgra);

  // Creates a color from BGRA channels in a uint format. Consumers should make
  // sure the memory order of the uint32_t is always BGRA from left to right, no
  // matter the system endian. This function creates an equivalent RgbaColor
  // instance from the ToUInt32() result of another RgbaColor instance.
  explicit RgbaColor(uint32_t bgra);

  // Returns true if `this` and `right` is the same color.
  bool operator==(const RgbaColor& right) const;

  // Returns true if `this` and `right` are different colors.
  bool operator!=(const RgbaColor& right) const;

  uint32_t ToUInt32() const;

  uint8_t blue;
  uint8_t green;
  uint8_t red;
  uint8_t alpha;
};
static_assert(
    DesktopFrame::kBytesPerPixel == sizeof(RgbaColor),
    "A pixel in DesktopFrame should be safe to be represented by a RgbaColor");

}  // namespace webrtc

#endif  // MODULES_DESKTOP_CAPTURE_RGBA_COLOR_H_
