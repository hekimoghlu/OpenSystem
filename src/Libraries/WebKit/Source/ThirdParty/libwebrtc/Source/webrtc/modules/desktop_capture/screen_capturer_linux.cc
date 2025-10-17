/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 3, 2024.
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
#include <memory>

#include "modules/desktop_capture/desktop_capture_options.h"
#include "modules/desktop_capture/desktop_capturer.h"

#if defined(WEBRTC_USE_PIPEWIRE)
#include "modules/desktop_capture/linux/wayland/base_capturer_pipewire.h"
#endif  // defined(WEBRTC_USE_PIPEWIRE)

#if defined(WEBRTC_USE_X11)
#include "modules/desktop_capture/linux/x11/screen_capturer_x11.h"
#endif  // defined(WEBRTC_USE_X11)

namespace webrtc {

// static
std::unique_ptr<DesktopCapturer> DesktopCapturer::CreateRawScreenCapturer(
    const DesktopCaptureOptions& options) {
#if defined(WEBRTC_USE_PIPEWIRE)
  if (options.allow_pipewire() && BaseCapturerPipeWire::IsSupported()) {
    return std::make_unique<BaseCapturerPipeWire>(options,
                                                  CaptureType::kScreen);
  }
#endif  // defined(WEBRTC_USE_PIPEWIRE)

#if defined(WEBRTC_USE_X11)
  if (!DesktopCapturer::IsRunningUnderWayland())
    return ScreenCapturerX11::CreateRawScreenCapturer(options);
#endif  // defined(WEBRTC_USE_X11)

  return nullptr;
}

}  // namespace webrtc
