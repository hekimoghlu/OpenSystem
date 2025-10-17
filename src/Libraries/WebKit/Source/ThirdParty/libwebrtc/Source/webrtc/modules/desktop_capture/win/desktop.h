/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 24, 2024.
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
#ifndef MODULES_DESKTOP_CAPTURE_WIN_DESKTOP_H_
#define MODULES_DESKTOP_CAPTURE_WIN_DESKTOP_H_

#include <windows.h>

#include <string>

#include "rtc_base/system/rtc_export.h"

namespace webrtc {

class RTC_EXPORT Desktop {
 public:
  ~Desktop();

  Desktop(const Desktop&) = delete;
  Desktop& operator=(const Desktop&) = delete;

  // Returns the name of the desktop represented by the object. Return false if
  // quering the name failed for any reason.
  bool GetName(std::wstring* desktop_name_out) const;

  // Returns true if `other` has the same name as this desktop. Returns false
  // in any other case including failing Win32 APIs and uninitialized desktop
  // handles.
  bool IsSame(const Desktop& other) const;

  // Assigns the desktop to the current thread. Returns false is the operation
  // failed for any reason.
  bool SetThreadDesktop() const;

  // Returns the desktop by its name or NULL if an error occurs.
  static Desktop* GetDesktop(const wchar_t* desktop_name);

  // Returns the desktop currently receiving user input or NULL if an error
  // occurs.
  static Desktop* GetInputDesktop();

  // Returns the desktop currently assigned to the calling thread or NULL if
  // an error occurs.
  static Desktop* GetThreadDesktop();

 private:
  Desktop(HDESK desktop, bool own);

  // The desktop handle.
  HDESK desktop_;

  // True if `desktop_` must be closed on teardown.
  bool own_;
};

}  // namespace webrtc

#endif  // MODULES_DESKTOP_CAPTURE_WIN_DESKTOP_H_
