/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 23, 2024.
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
#include <utility>

#include "modules/desktop_capture/blank_detector_desktop_capturer_wrapper.h"
#include "modules/desktop_capture/desktop_capture_options.h"
#include "modules/desktop_capture/desktop_capturer.h"
#include "modules/desktop_capture/fallback_desktop_capturer_wrapper.h"
#include "modules/desktop_capture/rgba_color.h"
#include "modules/desktop_capture/win/screen_capturer_win_directx.h"
#include "modules/desktop_capture/win/screen_capturer_win_gdi.h"

namespace webrtc {

namespace {

std::unique_ptr<DesktopCapturer> CreateScreenCapturerWinDirectx(
    const DesktopCaptureOptions& options) {
  std::unique_ptr<DesktopCapturer> capturer(
      new ScreenCapturerWinDirectx(options));
  capturer.reset(new BlankDetectorDesktopCapturerWrapper(
      std::move(capturer), RgbaColor(0, 0, 0, 0)));
  return capturer;
}

}  // namespace

// static
std::unique_ptr<DesktopCapturer> DesktopCapturer::CreateRawScreenCapturer(
    const DesktopCaptureOptions& options) {
  // Default capturer if no options are enabled is GDI.
  std::unique_ptr<DesktopCapturer> capturer(new ScreenCapturerWinGdi(options));

  // If DirectX is enabled use it as main capturer with GDI as fallback.
  if (options.allow_directx_capturer()) {
    // `dxgi_duplicator_controller` should be alive in this scope to ensure it
    // won't unload DxgiDuplicatorController.
    auto dxgi_duplicator_controller = DxgiDuplicatorController::Instance();
    if (ScreenCapturerWinDirectx::IsSupported()) {
      capturer.reset(new FallbackDesktopCapturerWrapper(
          CreateScreenCapturerWinDirectx(options), std::move(capturer)));
      return capturer;
    }
  }

  // Use GDI as default capturer without any fallback solution.
  return capturer;
}

}  // namespace webrtc
