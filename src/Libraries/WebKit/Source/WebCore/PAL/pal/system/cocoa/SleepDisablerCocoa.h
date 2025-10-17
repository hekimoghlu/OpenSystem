/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 23, 2023.
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
#pragma once

#if PLATFORM(COCOA)

#include <pal/system/SleepDisabler.h>
#include <wtf/RefCounter.h>

namespace PAL {

#if PLATFORM(IOS_FAMILY)
enum ScreenSleepDisablerCounterType { };
typedef RefCounter<ScreenSleepDisablerCounterType> ScreenSleepDisablerCounter;
typedef ScreenSleepDisablerCounter::Token ScreenSleepDisablerCounterToken;
#endif

class SleepDisablerCocoa : public SleepDisabler {
public:
    explicit SleepDisablerCocoa(const String&, Type);
    virtual ~SleepDisablerCocoa();

#if PLATFORM(IOS_FAMILY)
    PAL_EXPORT static void setScreenWakeLockHandler(Function<bool(bool shouldKeepScreenAwake)>&&);
#endif

private:
    void takeScreenSleepDisablingAssertion(const String& reason);
    void takeSystemSleepDisablingAssertion(const String& reason);

    uint32_t m_sleepAssertion { 0 };
#if PLATFORM(IOS_FAMILY)
    ScreenSleepDisablerCounterToken m_screenSleepDisablerToken;
#endif
};

} // namespace PAL

#endif // PLATFORM(COCOA)
