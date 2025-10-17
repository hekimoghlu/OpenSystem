/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 19, 2022.
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
#include "modules/desktop_capture/win/screen_capturer_win_directx.h"

#include <string>
#include <vector>

#include "modules/desktop_capture/desktop_capturer.h"
#include "test/gtest.h"

namespace webrtc {

// This test cannot ensure GetScreenListFromDeviceNames() won't reorder the
// devices in its output, since the device name is missing.
TEST(ScreenCaptureUtilsTest, GetScreenListFromDeviceNamesAndGetIndex) {
  const std::vector<std::string> device_names = {
      "\\\\.\\DISPLAY0",
      "\\\\.\\DISPLAY1",
      "\\\\.\\DISPLAY2",
  };
  DesktopCapturer::SourceList screens;
  ASSERT_TRUE(ScreenCapturerWinDirectx::GetScreenListFromDeviceNames(
      device_names, &screens));
  ASSERT_EQ(device_names.size(), screens.size());

  for (size_t i = 0; i < screens.size(); i++) {
    ASSERT_EQ(ScreenCapturerWinDirectx::GetIndexFromScreenId(screens[i].id,
                                                             device_names),
              static_cast<int>(i));
  }
}

}  // namespace webrtc
