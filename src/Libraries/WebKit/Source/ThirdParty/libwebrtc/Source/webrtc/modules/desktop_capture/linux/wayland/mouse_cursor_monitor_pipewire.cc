/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 28, 2024.
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
#include "modules/desktop_capture/linux/wayland/mouse_cursor_monitor_pipewire.h"

#include <utility>

#include "modules/desktop_capture/desktop_capture_options.h"
#include "modules/desktop_capture/desktop_capturer.h"
#include "rtc_base/checks.h"
#include "rtc_base/logging.h"

namespace webrtc {

MouseCursorMonitorPipeWire::MouseCursorMonitorPipeWire(
    const DesktopCaptureOptions& options)
    : options_(options) {
  sequence_checker_.Detach();
}

MouseCursorMonitorPipeWire::~MouseCursorMonitorPipeWire() {}

void MouseCursorMonitorPipeWire::Init(Callback* callback, Mode mode) {
  RTC_DCHECK_RUN_ON(&sequence_checker_);
  RTC_DCHECK(!callback_);
  RTC_DCHECK(callback);

  callback_ = callback;
  mode_ = mode;
}

void MouseCursorMonitorPipeWire::Capture() {
  RTC_DCHECK_RUN_ON(&sequence_checker_);
  RTC_DCHECK(callback_);

  std::optional<DesktopVector> mouse_cursor_position =
      options_.screencast_stream()->CaptureCursorPosition();
  // Invalid cursor or position
  if (!mouse_cursor_position) {
    callback_->OnMouseCursor(nullptr);
    return;
  }

  std::unique_ptr<MouseCursor> mouse_cursor =
      options_.screencast_stream()->CaptureCursor();

  if (mouse_cursor && mouse_cursor->image()->data()) {
    callback_->OnMouseCursor(mouse_cursor.release());
  }

  if (mode_ == SHAPE_AND_POSITION) {
    callback_->OnMouseCursorPosition(mouse_cursor_position.value());
  }
}

}  // namespace webrtc
