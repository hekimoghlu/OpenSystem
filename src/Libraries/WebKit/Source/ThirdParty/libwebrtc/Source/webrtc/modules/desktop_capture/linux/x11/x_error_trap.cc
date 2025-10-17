/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 30, 2023.
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
#include "modules/desktop_capture/linux/x11/x_error_trap.h"

#include <stddef.h>

#include <atomic>

#include "rtc_base/checks.h"

namespace webrtc {

namespace {

static int g_last_xserver_error_code = 0;
static std::atomic<Display*> g_display_for_error_handler = nullptr;

Mutex* AcquireMutex() {
  static Mutex* mutex = new Mutex();
  return mutex;
}

int XServerErrorHandler(Display* display, XErrorEvent* error_event) {
  RTC_DCHECK_EQ(display, g_display_for_error_handler.load());
  g_last_xserver_error_code = error_event->error_code;
  return 0;
}

}  // namespace

XErrorTrap::XErrorTrap(Display* display) : mutex_lock_(AcquireMutex()) {
  // We don't expect this class to be used in a nested fashion so therefore
  // g_display_for_error_handler should never be valid here.
  RTC_DCHECK(!g_display_for_error_handler.load());
  RTC_DCHECK(display);
  g_display_for_error_handler.store(display);
  g_last_xserver_error_code = 0;
  original_error_handler_ = XSetErrorHandler(&XServerErrorHandler);
}

int XErrorTrap::GetLastErrorAndDisable() {
  g_display_for_error_handler.store(nullptr);
  XSetErrorHandler(original_error_handler_);
  return g_last_xserver_error_code;
}

XErrorTrap::~XErrorTrap() {
  if (g_display_for_error_handler.load() != nullptr)
    GetLastErrorAndDisable();
}

}  // namespace webrtc
