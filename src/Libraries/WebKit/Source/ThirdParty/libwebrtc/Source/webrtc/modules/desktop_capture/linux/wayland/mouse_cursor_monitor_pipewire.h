/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 28, 2025.
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
#ifndef MODULES_DESKTOP_CAPTURE_LINUX_WAYLAND_MOUSE_CURSOR_MONITOR_PIPEWIRE_H_
#define MODULES_DESKTOP_CAPTURE_LINUX_WAYLAND_MOUSE_CURSOR_MONITOR_PIPEWIRE_H_

#include <memory>

#include "api/scoped_refptr.h"
#include "api/sequence_checker.h"
#include "modules/desktop_capture/desktop_capture_options.h"
#include "modules/desktop_capture/desktop_capture_types.h"
#include "modules/desktop_capture/linux/wayland/shared_screencast_stream.h"
#include "modules/desktop_capture/mouse_cursor.h"
#include "modules/desktop_capture/mouse_cursor_monitor.h"
#include "rtc_base/system/no_unique_address.h"

namespace webrtc {

class MouseCursorMonitorPipeWire : public MouseCursorMonitor {
 public:
  explicit MouseCursorMonitorPipeWire(const DesktopCaptureOptions& options);
  ~MouseCursorMonitorPipeWire() override;

  // MouseCursorMonitor:
  void Init(Callback* callback, Mode mode) override;
  void Capture() override;

  DesktopCaptureOptions options_ RTC_GUARDED_BY(sequence_checker_);
  Callback* callback_ RTC_GUARDED_BY(sequence_checker_) = nullptr;
  Mode mode_ RTC_GUARDED_BY(sequence_checker_) = SHAPE_AND_POSITION;
  RTC_NO_UNIQUE_ADDRESS SequenceChecker sequence_checker_;
};

}  // namespace webrtc

#endif  // MODULES_DESKTOP_CAPTURE_LINUX_WAYLAND_MOUSE_CURSOR_MONITOR_PIPEWIRE_H_
