/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 7, 2024.
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
#ifndef MODULES_DESKTOP_CAPTURE_SCREEN_DRAWER_H_
#define MODULES_DESKTOP_CAPTURE_SCREEN_DRAWER_H_

#include "modules/desktop_capture/desktop_capture_types.h"
#include "modules/desktop_capture/desktop_geometry.h"
#include "modules/desktop_capture/rgba_color.h"

namespace webrtc {

// A cross-process lock to ensure only one ScreenDrawer can be used at a certain
// time.
class ScreenDrawerLock {
 public:
  virtual ~ScreenDrawerLock();

  static std::unique_ptr<ScreenDrawerLock> Create();

 protected:
  ScreenDrawerLock();
};

// A set of basic platform dependent functions to draw various shapes on the
// screen.
class ScreenDrawer {
 public:
  // Creates a ScreenDrawer for the current platform, returns nullptr if no
  // ScreenDrawer implementation available.
  // If the implementation cannot guarantee two ScreenDrawer instances won't
  // impact each other, this function may block current thread until another
  // ScreenDrawer has been destroyed.
  static std::unique_ptr<ScreenDrawer> Create();

  ScreenDrawer();
  virtual ~ScreenDrawer();

  // Returns the region inside which DrawRectangle() function are expected to
  // work, in capturer coordinates (assuming ScreenCapturer::SelectScreen has
  // not been called). This region may exclude regions of the screen reserved by
  // the OS for things like menu bars or app launchers. The DesktopRect is in
  // system coordinate, i.e. the primary monitor always starts from (0, 0).
  virtual DesktopRect DrawableRegion() = 0;

  // Draws a rectangle to cover `rect` with `color`. Note, rect.bottom() and
  // rect.right() two lines are not included. The part of `rect` which is out of
  // DrawableRegion() will be ignored.
  virtual void DrawRectangle(DesktopRect rect, RgbaColor color) = 0;

  // Clears all content on the screen by filling the area with black.
  virtual void Clear() = 0;

  // Blocks current thread until OS finishes previous DrawRectangle() actions.
  // ScreenCapturer should be able to capture the changes after this function
  // finish.
  virtual void WaitForPendingDraws() = 0;

  // Returns true if incomplete shapes previous actions required may be drawn on
  // the screen after a WaitForPendingDraws() call. i.e. Though the complete
  // shapes will eventually be drawn on the screen, due to some OS limitations,
  // these shapes may be partially appeared sometimes.
  virtual bool MayDrawIncompleteShapes() = 0;

  // Returns the id of the drawer window. This function returns kNullWindowId if
  // the implementation does not draw on a window of the system.
  virtual WindowId window_id() const = 0;
};

}  // namespace webrtc

#endif  // MODULES_DESKTOP_CAPTURE_SCREEN_DRAWER_H_
