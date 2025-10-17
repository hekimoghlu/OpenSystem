/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 11, 2025.
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
#import "WebProcessProxy.h"

#if PLATFORM(IOS_FAMILY)

#import "APIUIClient.h"
#import "AccessibilitySupportSPI.h"
#import "WKFullKeyboardAccessWatcher.h"
#import "WKMouseDeviceObserver.h"
#import "WKStylusDeviceObserver.h"
#import "WebPageProxy.h"
#import "WebProcessMessages.h"
#import "WebProcessPool.h"
#import <pal/system/cocoa/SleepDisablerCocoa.h>
#import <wtf/BlockPtr.h>
#import <wtf/cocoa/RuntimeApplicationChecksCocoa.h>

namespace WebKit {

void WebProcessProxy::platformInitialize()
{
#if HAVE(MOUSE_DEVICE_OBSERVATION)
    [[WKMouseDeviceObserver sharedInstance] start];
#endif
#if HAVE(STYLUS_DEVICE_OBSERVATION)
    [[WKStylusDeviceObserver sharedInstance] start];
#endif

    static bool didSetScreenWakeLockHandler = false;
    if (!didSetScreenWakeLockHandler) {
        didSetScreenWakeLockHandler = true;
        PAL::SleepDisablerCocoa::setScreenWakeLockHandler([](bool shouldKeepScreenAwake) {
            RefPtr<WebPageProxy> visiblePage;
            for (auto&& page : globalPageMap().values()) {
                if (!visiblePage)
                    visiblePage = page.ptr();
                else if (page->isViewVisible()) {
                    visiblePage = page.ptr();
                    break;
                }
            }
            if (!visiblePage) {
                ASSERT_NOT_REACHED();
                return false;
            }
            return visiblePage->uiClient().setShouldKeepScreenAwake(shouldKeepScreenAwake);
        });
    }

    throttler().setAllowsActivities(!m_processPool->processesShouldSuspend());
}

void WebProcessProxy::platformDestroy()
{
#if HAVE(MOUSE_DEVICE_OBSERVATION)
    [[WKMouseDeviceObserver sharedInstance] stop];
#endif
#if HAVE(STYLUS_DEVICE_OBSERVATION)
    [[WKStylusDeviceObserver sharedInstance] stop];
#endif
}

bool WebProcessProxy::fullKeyboardAccessEnabled()
{
#if ENABLE(FULL_KEYBOARD_ACCESS)
    return [WKFullKeyboardAccessWatcher fullKeyboardAccessEnabled];
#else
    return NO;
#endif
}

} // namespace WebKit

#endif // PLATFORM(IOS_FAMILY)
