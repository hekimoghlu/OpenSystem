/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 1, 2023.
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
#include "modules/desktop_capture/linux/x11/shared_x_display.h"

#include <X11/Xlib.h>
#include <X11/extensions/XTest.h>

#include <algorithm>

#include "absl/strings/string_view.h"
#include "rtc_base/checks.h"
#include "rtc_base/logging.h"

namespace webrtc {

SharedXDisplay::SharedXDisplay(Display* display) : display_(display) {
  RTC_DCHECK(display_);
}

SharedXDisplay::~SharedXDisplay() {
  RTC_DCHECK(event_handlers_.empty());
  XCloseDisplay(display_);
}

// static
rtc::scoped_refptr<SharedXDisplay> SharedXDisplay::Create(
    absl::string_view display_name) {
  Display* display = XOpenDisplay(
      display_name.empty() ? NULL : std::string(display_name).c_str());
  if (!display) {
    RTC_LOG(LS_ERROR) << "Unable to open display";
    return nullptr;
  }
  return rtc::scoped_refptr<SharedXDisplay>(new SharedXDisplay(display));
}

// static
rtc::scoped_refptr<SharedXDisplay> SharedXDisplay::CreateDefault() {
  return Create(std::string());
}

void SharedXDisplay::AddEventHandler(int type, XEventHandler* handler) {
  MutexLock lock(&mutex_);
  event_handlers_[type].push_back(handler);
}

void SharedXDisplay::RemoveEventHandler(int type, XEventHandler* handler) {
  MutexLock lock(&mutex_);
  EventHandlersMap::iterator handlers = event_handlers_.find(type);
  if (handlers == event_handlers_.end())
    return;

  std::vector<XEventHandler*>::iterator new_end =
      std::remove(handlers->second.begin(), handlers->second.end(), handler);
  handlers->second.erase(new_end, handlers->second.end());

  // Check if no handlers left for this event.
  if (handlers->second.empty())
    event_handlers_.erase(handlers);
}

void SharedXDisplay::ProcessPendingXEvents() {
  // Hold reference to `this` to prevent it from being destroyed while
  // processing events.
  rtc::scoped_refptr<SharedXDisplay> self(this);

  // Protect access to `event_handlers_` after incrementing the refcount for
  // `this` to ensure the instance is still valid when the lock is acquired.
  MutexLock lock(&mutex_);

  // Find the number of events that are outstanding "now."  We don't just loop
  // on XPending because we want to guarantee this terminates.
  int events_to_process = XPending(display());
  XEvent e;

  for (int i = 0; i < events_to_process; i++) {
    XNextEvent(display(), &e);
    EventHandlersMap::iterator handlers = event_handlers_.find(e.type);
    if (handlers == event_handlers_.end())
      continue;
    for (std::vector<XEventHandler*>::iterator it = handlers->second.begin();
         it != handlers->second.end(); ++it) {
      if ((*it)->HandleXEvent(e))
        break;
    }
  }
}

void SharedXDisplay::IgnoreXServerGrabs() {
  int test_event_base = 0;
  int test_error_base = 0;
  int major = 0;
  int minor = 0;
  if (XTestQueryExtension(display(), &test_event_base, &test_error_base, &major,
                          &minor)) {
    XTestGrabControl(display(), true);
  }
}

}  // namespace webrtc
