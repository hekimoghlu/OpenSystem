/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 24, 2023.
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
#import "config.h"
#import "SleepDisablerCocoa.h"

#if PLATFORM(IOS_FAMILY)
#import "Logging.h"
#import <pal/spi/ios/UIKitSPI.h>
#import <wtf/NeverDestroyed.h>

#import <pal/ios/UIKitSoftLink.h>

namespace PAL {

class ScreenSleepDisabler {
public:
    static ScreenSleepDisabler& shared()
    {
        static MainThreadNeverDestroyed<ScreenSleepDisabler> screenSleepDisabler;
        return screenSleepDisabler;
    }

    ScreenSleepDisablerCounterToken takeAssertion()
    {
        return m_screenSleepDisablerCount.count();
    }

    void setScreenWakeLockHandler(Function<bool(bool shouldKeepScreenAwake)>&& screenWakeLockHandler)
    {
        m_screenWakeLockHandler = WTFMove(screenWakeLockHandler);
    }

private:
    friend NeverDestroyed<ScreenSleepDisabler, WTF::MainThreadAccessTraits>;

    ScreenSleepDisabler()
        : m_screenSleepDisablerCount([this](RefCounterEvent) { updateState(); })
    { }

    void updateState()
    {
        if (m_screenSleepDisablerCount.value() > 1)
            return;

        bool shouldKeepScreenAwake = !!m_screenSleepDisablerCount.value();
        RELEASE_LOG(Media, "ScreenSleepDisabler::updateState() shouldKeepScreenAwake=%d", shouldKeepScreenAwake);
        ensureOnMainRunLoop([this, shouldKeepScreenAwake] {
            if (m_screenWakeLockHandler && m_screenWakeLockHandler(shouldKeepScreenAwake))
                return;
            [[PAL::getUIApplicationClass() sharedApplication] _setIdleTimerDisabled:shouldKeepScreenAwake forReason:@"WebKit SleepDisabler"];
        });
    }
    ScreenSleepDisablerCounter m_screenSleepDisablerCount;
    Function<bool(bool shouldKeepScreenAwake)> m_screenWakeLockHandler;
};

void SleepDisablerCocoa::takeScreenSleepDisablingAssertion(const String&)
{
    m_screenSleepDisablerToken = ScreenSleepDisabler::shared().takeAssertion();
}

void SleepDisablerCocoa::setScreenWakeLockHandler(Function<bool(bool shouldKeepScreenAwake)>&& screenWakeLockHandler)
{
    ScreenSleepDisabler::shared().setScreenWakeLockHandler(WTFMove(screenWakeLockHandler));
}

} // namespace PAL

#endif // PLATFORM(IOS_FAMILY)
