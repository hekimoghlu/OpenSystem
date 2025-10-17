/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 11, 2024.
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
#ifndef MODULES_DESKTOP_CAPTURE_DESKTOP_CAPTURE_METADATA_H_
#define MODULES_DESKTOP_CAPTURE_DESKTOP_CAPTURE_METADATA_H_

#if defined(WEBRTC_USE_GIO)
#include "modules/portal/xdg_session_details.h"
#endif  // defined(WEBRTC_USE_GIO)

namespace webrtc {

// Container for the metadata associated with a desktop capturer.
struct DesktopCaptureMetadata {
#if defined(WEBRTC_USE_GIO)
  // Details about the XDG desktop session handle (used by wayland
  // implementation in remoting)
  xdg_portal::SessionDetails session_details;
#endif  // defined(WEBRTC_USE_GIO)
};

}  // namespace webrtc

#endif  // MODULES_DESKTOP_CAPTURE_DESKTOP_CAPTURE_METADATA_H_
