/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 12, 2025.
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
#include "rtc_base/win/windows_version.h"

#include "rtc_base/gunit.h"
#include "rtc_base/logging.h"

namespace rtc {
namespace rtc_win {
namespace {

void MethodSupportedOnWin10AndLater() {
  RTC_DLOG(LS_INFO) << "MethodSupportedOnWin10AndLater";
}

void MethodNotSupportedOnWin10AndLater() {
  RTC_DLOG(LS_INFO) << "MethodNotSupportedOnWin10AndLater";
}

// Use global GetVersion() and use it in a way a user would typically use it
// when checking for support of a certain API:
// "if (rtc_win::GetVersion() < VERSION_WIN10) ...".
TEST(WindowsVersion, GetVersionGlobalScopeAccessor) {
  if (GetVersion() < VERSION_WIN10) {
    MethodNotSupportedOnWin10AndLater();
  } else {
    MethodSupportedOnWin10AndLater();
  }
}

TEST(WindowsVersion, ProcessorModelName) {
  std::string name = OSInfo::GetInstance()->processor_model_name();
  EXPECT_FALSE(name.empty());
  RTC_DLOG(LS_INFO) << "processor_model_name: " << name;
}

}  // namespace
}  // namespace rtc_win
}  // namespace rtc
