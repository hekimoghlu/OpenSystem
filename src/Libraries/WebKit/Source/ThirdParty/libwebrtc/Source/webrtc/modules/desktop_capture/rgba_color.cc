/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 18, 2025.
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
#include "modules/desktop_capture/rgba_color.h"

#include "rtc_base/system/arch.h"

namespace webrtc {

namespace {

bool AlphaEquals(uint8_t i, uint8_t j) {
  // On Linux and Windows 8 or early version, '0' was returned for alpha channel
  // from capturer APIs, on Windows 10, '255' was returned. So a workaround is
  // to treat 0 as 255.
  return i == j || ((i == 0 || i == 255) && (j == 0 || j == 255));
}

}  // namespace

RgbaColor::RgbaColor(uint8_t blue, uint8_t green, uint8_t red, uint8_t alpha) {
  this->blue = blue;
  this->green = green;
  this->red = red;
  this->alpha = alpha;
}

RgbaColor::RgbaColor(uint8_t blue, uint8_t green, uint8_t red)
    : RgbaColor(blue, green, red, 0xff) {}

RgbaColor::RgbaColor(const uint8_t* bgra)
    : RgbaColor(bgra[0], bgra[1], bgra[2], bgra[3]) {}

RgbaColor::RgbaColor(uint32_t bgra)
    : RgbaColor(reinterpret_cast<uint8_t*>(&bgra)) {}

bool RgbaColor::operator==(const RgbaColor& right) const {
  return blue == right.blue && green == right.green && red == right.red &&
         AlphaEquals(alpha, right.alpha);
}

bool RgbaColor::operator!=(const RgbaColor& right) const {
  return !(*this == right);
}

uint32_t RgbaColor::ToUInt32() const {
#if defined(WEBRTC_ARCH_LITTLE_ENDIAN)
  return blue | (green << 8) | (red << 16) | (alpha << 24);
#else
  return (blue << 24) | (green << 16) | (red << 8) | alpha;
#endif
}

}  // namespace webrtc
