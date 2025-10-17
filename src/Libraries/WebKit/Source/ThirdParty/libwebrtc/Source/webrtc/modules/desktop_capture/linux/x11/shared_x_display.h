/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 31, 2025.
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
#ifndef MODULES_DESKTOP_CAPTURE_LINUX_X11_SHARED_X_DISPLAY_H_
#define MODULES_DESKTOP_CAPTURE_LINUX_X11_SHARED_X_DISPLAY_H_

#include <map>
#include <vector>

#include "absl/strings/string_view.h"
#include "api/ref_counted_base.h"
#include "api/scoped_refptr.h"
#include "rtc_base/synchronization/mutex.h"
#include "rtc_base/system/rtc_export.h"
#include "rtc_base/thread_annotations.h"

// Including Xlib.h will involve evil defines (Bool, Status, True, False), which
// easily conflict with other headers.
typedef struct _XDisplay Display;
typedef union _XEvent XEvent;

namespace webrtc {

// A ref-counted object to store XDisplay connection.
class RTC_EXPORT SharedXDisplay
    : public rtc::RefCountedNonVirtual<SharedXDisplay> {
 public:
  class XEventHandler {
   public:
    virtual ~XEventHandler() {}

    // Processes XEvent. Returns true if the event has been handled.
    virtual bool HandleXEvent(const XEvent& event) = 0;
  };

  // Creates a new X11 Display for the `display_name`. NULL is returned if X11
  // connection failed. Equivalent to CreateDefault() when `display_name` is
  // empty.
  static rtc::scoped_refptr<SharedXDisplay> Create(
      absl::string_view display_name);

  // Creates X11 Display connection for the default display (e.g. specified in
  // DISPLAY). NULL is returned if X11 connection failed.
  static rtc::scoped_refptr<SharedXDisplay> CreateDefault();

  Display* display() { return display_; }

  // Adds a new event `handler` for XEvent's of `type`.
  void AddEventHandler(int type, XEventHandler* handler);

  // Removes event `handler` added using `AddEventHandler`. Doesn't do anything
  // if `handler` is not registered.
  void RemoveEventHandler(int type, XEventHandler* handler);

  // Processes pending XEvents, calling corresponding event handlers.
  void ProcessPendingXEvents();

  void IgnoreXServerGrabs();

  ~SharedXDisplay();

  SharedXDisplay(const SharedXDisplay&) = delete;
  SharedXDisplay& operator=(const SharedXDisplay&) = delete;

 protected:
  // Takes ownership of `display`.
  explicit SharedXDisplay(Display* display);

 private:
  typedef std::map<int, std::vector<XEventHandler*> > EventHandlersMap;

  Display* display_;

  Mutex mutex_;

  EventHandlersMap event_handlers_ RTC_GUARDED_BY(mutex_);
};

}  // namespace webrtc

#endif  // MODULES_DESKTOP_CAPTURE_LINUX_X11_SHARED_X_DISPLAY_H_
