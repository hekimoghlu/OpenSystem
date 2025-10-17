/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 7, 2022.
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
#include "modules/desktop_capture/win/screen_capture_utils.h"

#include <string>
#include <vector>

#include "modules/desktop_capture/desktop_capture_types.h"
#include "modules/desktop_capture/desktop_capturer.h"
#include "rtc_base/logging.h"
#include "test/gtest.h"

namespace webrtc {

TEST(ScreenCaptureUtilsTest, GetScreenList) {
  DesktopCapturer::SourceList screens;
  std::vector<std::string> device_names;

  ASSERT_TRUE(GetScreenList(&screens));
  screens.clear();
  ASSERT_TRUE(GetScreenList(&screens, &device_names));

  ASSERT_EQ(screens.size(), device_names.size());
}

TEST(ScreenCaptureUtilsTest, DeviceIndexToHmonitor) {
  DesktopCapturer::SourceList screens;
  ASSERT_TRUE(GetScreenList(&screens));
  if (screens.empty()) {
    RTC_LOG(LS_INFO)
        << "Skip ScreenCaptureUtilsTest on systems with no monitors.";
    GTEST_SKIP();
  }

  HMONITOR hmonitor;
  ASSERT_TRUE(GetHmonitorFromDeviceIndex(screens[0].id, &hmonitor));
  ASSERT_TRUE(IsMonitorValid(hmonitor));
}

TEST(ScreenCaptureUtilsTest, FullScreenDeviceIndexToHmonitor) {
  if (!HasActiveDisplay()) {
    RTC_LOG(LS_INFO)
        << "Skip ScreenCaptureUtilsTest on systems with no monitors.";
    GTEST_SKIP();
  }

  HMONITOR hmonitor;
  ASSERT_TRUE(GetHmonitorFromDeviceIndex(kFullDesktopScreenId, &hmonitor));
  ASSERT_EQ(hmonitor, static_cast<HMONITOR>(0));
  ASSERT_TRUE(IsMonitorValid(hmonitor));
}

TEST(ScreenCaptureUtilsTest, NoMonitors) {
  if (HasActiveDisplay()) {
    RTC_LOG(LS_INFO) << "Skip ScreenCaptureUtilsTest designed specifically for "
                        "systems with no monitors";
    GTEST_SKIP();
  }

  HMONITOR hmonitor;
  ASSERT_TRUE(GetHmonitorFromDeviceIndex(kFullDesktopScreenId, &hmonitor));
  ASSERT_EQ(hmonitor, static_cast<HMONITOR>(0));

  // The monitor should be invalid since the system has no attached displays.
  ASSERT_FALSE(IsMonitorValid(hmonitor));
}

TEST(ScreenCaptureUtilsTest, InvalidDeviceIndexToHmonitor) {
  HMONITOR hmonitor;
  ASSERT_FALSE(GetHmonitorFromDeviceIndex(kInvalidScreenId, &hmonitor));
}

}  // namespace webrtc
