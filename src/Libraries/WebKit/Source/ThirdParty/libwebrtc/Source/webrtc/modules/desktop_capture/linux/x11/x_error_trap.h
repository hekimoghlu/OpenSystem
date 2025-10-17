/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 10, 2022.
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
#ifndef MODULES_DESKTOP_CAPTURE_LINUX_X11_X_ERROR_TRAP_H_
#define MODULES_DESKTOP_CAPTURE_LINUX_X11_X_ERROR_TRAP_H_

#include <X11/Xlib.h>

#include "rtc_base/synchronization/mutex.h"

namespace webrtc {

// Helper class that registers an X Window error handler. Caller can use
// GetLastErrorAndDisable() to get the last error that was caught, if any.
class XErrorTrap {
 public:
  explicit XErrorTrap(Display* display);

  XErrorTrap(const XErrorTrap&) = delete;
  XErrorTrap& operator=(const XErrorTrap&) = delete;

  ~XErrorTrap();

  // Returns the last error if one was caught, otherwise 0. Also unregisters the
  // error handler and replaces it with `original_error_handler_`.
  int GetLastErrorAndDisable();

 private:
  MutexLock mutex_lock_;
  XErrorHandler original_error_handler_ = nullptr;
};

}  // namespace webrtc

#endif  // MODULES_DESKTOP_CAPTURE_LINUX_X11_X_ERROR_TRAP_H_
