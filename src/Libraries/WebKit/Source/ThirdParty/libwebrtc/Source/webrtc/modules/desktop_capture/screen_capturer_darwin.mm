/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 30, 2021.
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

#include "modules/desktop_capture/mac/screen_capturer_mac.h"
#include "modules/desktop_capture/mac/screen_capturer_sck.h"

namespace webrtc {

// static
std::unique_ptr<DesktopCapturer> DesktopCapturer::CreateRawScreenCapturer(
    const DesktopCaptureOptions& options) {
  if (!options.configuration_monitor()) {
    return nullptr;
  }

  if (options.allow_sck_capturer()) {
    // This will return nullptr on systems that don't support ScreenCaptureKit.
    std::unique_ptr<DesktopCapturer> sck_capturer = CreateScreenCapturerSck(options);
    if (sck_capturer) {
      return sck_capturer;
    }
  }

  auto capturer = std::make_unique<ScreenCapturerMac>(
      options.configuration_monitor(), options.detect_updated_region(), options.allow_iosurface());
  if (!capturer->Init()) {
    return nullptr;
  }

  return capturer;
}

}  // namespace webrtc
