/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 5, 2022.
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
#ifndef MODULES_DESKTOP_CAPTURE_FULL_SCREEN_WINDOW_DETECTOR_H_
#define MODULES_DESKTOP_CAPTURE_FULL_SCREEN_WINDOW_DETECTOR_H_

#include <memory>

#include "api/function_view.h"
#include "api/ref_counted_base.h"
#include "api/scoped_refptr.h"
#include "modules/desktop_capture/desktop_capturer.h"
#include "modules/desktop_capture/full_screen_application_handler.h"

namespace webrtc {

// This is a way to handle switch to full-screen mode for application in some
// specific cases:
// - Chrome on MacOS creates a new window in full-screen mode to
//   show a tab full-screen and minimizes the old window.
// - PowerPoint creates new windows in full-screen mode when user goes to
//   presentation mode (Slide Show Window, Presentation Window).
//
// To continue capturing in these cases, we try to find the new full-screen
// window using criteria provided by application specific
// FullScreenApplicationHandler.

class FullScreenWindowDetector
    : public rtc::RefCountedNonVirtual<FullScreenWindowDetector> {
 public:
  using ApplicationHandlerFactory =
      std::function<std::unique_ptr<FullScreenApplicationHandler>(
          DesktopCapturer::SourceId sourceId)>;

  FullScreenWindowDetector(
      ApplicationHandlerFactory application_handler_factory);

  FullScreenWindowDetector(const FullScreenWindowDetector&) = delete;
  FullScreenWindowDetector& operator=(const FullScreenWindowDetector&) = delete;

  // Returns the full-screen window in place of the original window if all the
  // criteria provided by FullScreenApplicationHandler are met, or 0 if no such
  // window found.
  DesktopCapturer::SourceId FindFullScreenWindow(
      DesktopCapturer::SourceId original_source_id);

  // The caller should call this function periodically, implementation will
  // update internal state no often than twice per second
  void UpdateWindowListIfNeeded(
      DesktopCapturer::SourceId original_source_id,
      rtc::FunctionView<bool(DesktopCapturer::SourceList*)> get_sources);

  static rtc::scoped_refptr<FullScreenWindowDetector>
  CreateFullScreenWindowDetector();

 protected:
  std::unique_ptr<FullScreenApplicationHandler> app_handler_;

 private:
  void CreateApplicationHandlerIfNeeded(DesktopCapturer::SourceId source_id);

  ApplicationHandlerFactory application_handler_factory_;

  int64_t last_update_time_ms_;
  DesktopCapturer::SourceId previous_source_id_;

  // Save the source id when we fail to create an instance of
  // CreateApplicationHandlerIfNeeded to avoid redundant attempt to do it again.
  DesktopCapturer::SourceId no_handler_source_id_;

  DesktopCapturer::SourceList window_list_;
};

}  // namespace webrtc

#endif  // MODULES_DESKTOP_CAPTURE_FULL_SCREEN_WINDOW_DETECTOR_H_
