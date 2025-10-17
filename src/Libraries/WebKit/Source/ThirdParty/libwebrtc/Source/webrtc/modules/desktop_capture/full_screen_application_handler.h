/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 2, 2022.
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
#ifndef MODULES_DESKTOP_CAPTURE_FULL_SCREEN_APPLICATION_HANDLER_H_
#define MODULES_DESKTOP_CAPTURE_FULL_SCREEN_APPLICATION_HANDLER_H_

#include <memory>

#include "modules/desktop_capture/desktop_capturer.h"

namespace webrtc {

// Base class for application specific handler to check criteria for switch to
// full-screen mode and find if possible the full-screen window to share.
// Supposed to be created and owned by platform specific
// FullScreenWindowDetector.
class FullScreenApplicationHandler {
 public:
  virtual ~FullScreenApplicationHandler() {}

  FullScreenApplicationHandler(const FullScreenApplicationHandler&) = delete;
  FullScreenApplicationHandler& operator=(const FullScreenApplicationHandler&) =
      delete;

  explicit FullScreenApplicationHandler(DesktopCapturer::SourceId sourceId);

  // Returns the full-screen window in place of the original window if all the
  // criteria are met, or 0 if no such window found.
  virtual DesktopCapturer::SourceId FindFullScreenWindow(
      const DesktopCapturer::SourceList& window_list,
      int64_t timestamp) const;

  // Returns source id of original window associated with
  // FullScreenApplicationHandler
  DesktopCapturer::SourceId GetSourceId() const;

 private:
  const DesktopCapturer::SourceId source_id_;
};

}  // namespace webrtc

#endif  // MODULES_DESKTOP_CAPTURE_FULL_SCREEN_APPLICATION_HANDLER_H_
