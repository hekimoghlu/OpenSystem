/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 18, 2023.
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
#ifndef MODULES_DESKTOP_CAPTURE_WIN_SCOPED_THREAD_DESKTOP_H_
#define MODULES_DESKTOP_CAPTURE_WIN_SCOPED_THREAD_DESKTOP_H_

#include <windows.h>

#include <memory>

#include "rtc_base/system/rtc_export.h"

namespace webrtc {

class Desktop;

class RTC_EXPORT ScopedThreadDesktop {
 public:
  ScopedThreadDesktop();
  ~ScopedThreadDesktop();

  ScopedThreadDesktop(const ScopedThreadDesktop&) = delete;
  ScopedThreadDesktop& operator=(const ScopedThreadDesktop&) = delete;

  // Returns true if `desktop` has the same desktop name as the currently
  // assigned desktop (if assigned) or as the initial desktop (if not assigned).
  // Returns false in any other case including failing Win32 APIs and
  // uninitialized desktop handles.
  bool IsSame(const Desktop& desktop);

  // Reverts the calling thread to use the initial desktop.
  void Revert();

  // Assigns `desktop` to be the calling thread. Returns true if the thread has
  // been switched to `desktop` successfully. Takes ownership of `desktop`.
  bool SetThreadDesktop(Desktop* desktop);

 private:
  // The desktop handle assigned to the calling thread by Set
  std::unique_ptr<Desktop> assigned_;

  // The desktop handle assigned to the calling thread at creation.
  std::unique_ptr<Desktop> initial_;
};

}  // namespace webrtc

#endif  // MODULES_DESKTOP_CAPTURE_WIN_SCOPED_THREAD_DESKTOP_H_
