/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 19, 2023.
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
#include "modules/desktop_capture/window_finder.h"

#include <stdint.h>

#include <memory>

#include "api/scoped_refptr.h"
#include "modules/desktop_capture/desktop_geometry.h"
#include "modules/desktop_capture/screen_drawer.h"
#include "rtc_base/logging.h"
#include "test/gtest.h"

#if defined(WEBRTC_USE_X11)
#include "modules/desktop_capture/linux/x11/shared_x_display.h"
#include "modules/desktop_capture/linux/x11/x_atom_cache.h"
#endif

#if defined(WEBRTC_WIN)
#include <windows.h>

#include "modules/desktop_capture/win/window_capture_utils.h"
#include "modules/desktop_capture/window_finder_win.h"
#endif

namespace webrtc {

namespace {

#if defined(WEBRTC_WIN)
// ScreenDrawerWin does not have a message loop, so it's unresponsive to user
// inputs. WindowFinderWin cannot detect this kind of unresponsive windows.
// Instead, console window is used to test WindowFinderWin.
// TODO(b/373792116): Reenable once flakiness is fixed.
TEST(WindowFinderTest, DISABLED_FindConsoleWindow) {
  // Creates a ScreenDrawer to avoid this test from conflicting with
  // ScreenCapturerIntegrationTest: both tests require its window to be in
  // foreground.
  //
  // In ScreenCapturer related tests, this is controlled by
  // ScreenDrawer, which has a global lock to ensure only one ScreenDrawer
  // window is active. So even we do not use ScreenDrawer for Windows test,
  // creating an instance can block ScreenCapturer related tests until this test
  // finishes.
  //
  // Usually the test framework should take care of this "isolated test"
  // requirement, but unfortunately WebRTC trybots do not support this.
  std::unique_ptr<ScreenDrawer> drawer = ScreenDrawer::Create();
  const int kMaxSize = 10000;
  // Enlarges current console window.
  system("mode 1000,1000");
  const HWND console_window = GetConsoleWindow();
  // Ensures that current console window is visible.
  ShowWindow(console_window, SW_MAXIMIZE);
  // Moves the window to the top-left of the display.
  if (!MoveWindow(console_window, 0, 0, kMaxSize, kMaxSize, true)) {
    FAIL() << "Failed to move window. Error code: " << GetLastError();
  }

  bool should_restore_notopmost =
      (GetWindowLong(console_window, GWL_EXSTYLE) & WS_EX_TOPMOST) == 0;

  // Brings console window to top.
  if (!SetWindowPos(console_window, HWND_TOPMOST, 0, 0, 0, 0,
                    SWP_NOMOVE | SWP_NOSIZE)) {
    FAIL() << "Failed to bring window to top. Error code: " << GetLastError();
  }
  if (!BringWindowToTop(console_window)) {
    FAIL() << "Failed second attempt to bring window to top. Error code: "
           << GetLastError();
  }

  bool success = false;
  WindowFinderWin finder;
  for (int i = 0; i < kMaxSize; i++) {
    const DesktopVector spot(i, i);
    const HWND id = reinterpret_cast<HWND>(finder.GetWindowUnderPoint(spot));

    if (id == console_window) {
      success = true;
      break;
    }
    RTC_LOG(LS_INFO) << "Expected window " << console_window
                     << ". Found window " << id;
  }
  if (should_restore_notopmost)
    SetWindowPos(console_window, HWND_NOTOPMOST, 0, 0, 0, 0,
                 SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE);

  if (!success)
    FAIL();
}

#else
TEST(WindowFinderTest, FindDrawerWindow) {
  WindowFinder::Options options;
#if defined(WEBRTC_USE_X11)
  std::unique_ptr<XAtomCache> cache;
  const auto shared_x_display = SharedXDisplay::CreateDefault();
  if (shared_x_display) {
    cache = std::make_unique<XAtomCache>(shared_x_display->display());
    options.cache = cache.get();
  }
#endif
  std::unique_ptr<WindowFinder> finder = WindowFinder::Create(options);
  if (!finder) {
    RTC_LOG(LS_WARNING)
        << "No WindowFinder implementation for current platform.";
    return;
  }

  std::unique_ptr<ScreenDrawer> drawer = ScreenDrawer::Create();
  if (!drawer) {
    RTC_LOG(LS_WARNING)
        << "No ScreenDrawer implementation for current platform.";
    return;
  }

  if (drawer->window_id() == kNullWindowId) {
    // TODO(zijiehe): WindowFinderTest can use a dedicated window without
    // relying on ScreenDrawer.
    RTC_LOG(LS_WARNING)
        << "ScreenDrawer implementation for current platform does "
           "create a window.";
    return;
  }

  // ScreenDrawer may not be able to bring the window to the top. So we test
  // several spots, at least one of them should succeed.
  const DesktopRect region = drawer->DrawableRegion();
  if (region.is_empty()) {
    RTC_LOG(LS_WARNING)
        << "ScreenDrawer::DrawableRegion() is too small for the "
           "WindowFinderTest.";
    return;
  }

  for (int i = 0; i < region.width(); i++) {
    const DesktopVector spot(
        region.left() + i, region.top() + i * region.height() / region.width());
    const WindowId id = finder->GetWindowUnderPoint(spot);
    if (id == drawer->window_id()) {
      return;
    }
  }

  FAIL();
}
#endif

TEST(WindowFinderTest, ShouldReturnNullWindowIfSpotIsOutOfScreen) {
  WindowFinder::Options options;
#if defined(WEBRTC_USE_X11)
  std::unique_ptr<XAtomCache> cache;
  const auto shared_x_display = SharedXDisplay::CreateDefault();
  if (shared_x_display) {
    cache = std::make_unique<XAtomCache>(shared_x_display->display());
    options.cache = cache.get();
  }
#endif
  std::unique_ptr<WindowFinder> finder = WindowFinder::Create(options);
  if (!finder) {
    RTC_LOG(LS_WARNING)
        << "No WindowFinder implementation for current platform.";
    return;
  }

  ASSERT_EQ(kNullWindowId,
            finder->GetWindowUnderPoint(DesktopVector(INT16_MAX, INT16_MAX)));
  ASSERT_EQ(kNullWindowId,
            finder->GetWindowUnderPoint(DesktopVector(INT16_MAX, INT16_MIN)));
  ASSERT_EQ(kNullWindowId,
            finder->GetWindowUnderPoint(DesktopVector(INT16_MIN, INT16_MAX)));
  ASSERT_EQ(kNullWindowId,
            finder->GetWindowUnderPoint(DesktopVector(INT16_MIN, INT16_MIN)));
}

}  // namespace

}  // namespace webrtc
