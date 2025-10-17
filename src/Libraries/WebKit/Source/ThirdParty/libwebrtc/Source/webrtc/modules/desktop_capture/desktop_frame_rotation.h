/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 3, 2023.
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
#ifndef MODULES_DESKTOP_CAPTURE_DESKTOP_FRAME_ROTATION_H_
#define MODULES_DESKTOP_CAPTURE_DESKTOP_FRAME_ROTATION_H_

#include "modules/desktop_capture/desktop_frame.h"
#include "modules/desktop_capture/desktop_geometry.h"

namespace webrtc {

// Represents the rotation of a DesktopFrame.
enum class Rotation {
  CLOCK_WISE_0,
  CLOCK_WISE_90,
  CLOCK_WISE_180,
  CLOCK_WISE_270,
};

// Rotates input DesktopFrame `source`, copies pixel in an unrotated rectangle
// `source_rect` into the target rectangle of another DesktopFrame `target`.
// Target rectangle here is the rotated `source_rect` plus `target_offset`.
// `rotation` specifies `source` to `target` rotation. `source_rect` is in
// `source` coordinate. `target_offset` is in `target` coordinate.
// This function triggers check failure if `source` does not cover the
// `source_rect`, or `target` does not cover the rotated `rect`.
void RotateDesktopFrame(const DesktopFrame& source,
                        const DesktopRect& source_rect,
                        const Rotation& rotation,
                        const DesktopVector& target_offset,
                        DesktopFrame* target);

// Returns a reverse rotation of `rotation`.
Rotation ReverseRotation(Rotation rotation);

// Returns a rotated DesktopSize of `size`.
DesktopSize RotateSize(DesktopSize size, Rotation rotation);

// Returns a rotated DesktopRect of `rect`. The `size` represents the size of
// the DesktopFrame which `rect` belongs in.
DesktopRect RotateRect(DesktopRect rect, DesktopSize size, Rotation rotation);

}  // namespace webrtc

#endif  // MODULES_DESKTOP_CAPTURE_DESKTOP_FRAME_ROTATION_H_
