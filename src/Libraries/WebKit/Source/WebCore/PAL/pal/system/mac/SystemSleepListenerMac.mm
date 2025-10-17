/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 10, 2022.
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
#import "SystemSleepListenerMac.h"

#if PLATFORM(MAC)

#import <AppKit/AppKit.h>
#import <wtf/MainThread.h>
#import <wtf/TZoneMallocInlines.h>

namespace PAL {

WTF_MAKE_TZONE_ALLOCATED_IMPL(SystemSleepListenerMac);

std::unique_ptr<SystemSleepListener> SystemSleepListener::create(Client& client)
{
    return std::unique_ptr<SystemSleepListener>(new SystemSleepListenerMac(client));
}

SystemSleepListenerMac::SystemSleepListenerMac(Client& client)
    : SystemSleepListener(client)
    , m_sleepObserver(nil)
    , m_wakeObserver(nil)
{
    NSNotificationCenter *center = [[NSWorkspace sharedWorkspace] notificationCenter];
    NSOperationQueue *queue = [NSOperationQueue mainQueue];

    WeakPtr weakThis { *this };

    m_sleepObserver = [center addObserverForName:NSWorkspaceWillSleepNotification object:nil queue:queue usingBlock:^(NSNotification *) {
        callOnMainThread([weakThis] {
            if (weakThis)
                weakThis->m_client.systemWillSleep();
        });
    }];

    m_wakeObserver = [center addObserverForName:NSWorkspaceDidWakeNotification object:nil queue:queue usingBlock:^(NSNotification *) {
        callOnMainThread([weakThis] {
            if (weakThis)
                weakThis->m_client.systemDidWake();
        });
    }];
}

SystemSleepListenerMac::~SystemSleepListenerMac()
{
    NSNotificationCenter* center = [[NSWorkspace sharedWorkspace] notificationCenter];
    [center removeObserver:m_sleepObserver];
    [center removeObserver:m_wakeObserver];
}

} // namespace PAL

#endif // PLATFORM(MAC)
