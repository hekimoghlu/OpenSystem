/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 4, 2024.
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

#if PLATFORM(MAC)
#import "PowerObserverMac.h"
#import <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(PowerObserver);

PowerObserver::PowerObserver(Function<void()>&& powerOnHander)
    : m_powerOnHander(WTFMove(powerOnHander))
    , m_powerConnection(0)
    , m_notificationPort(nullptr)
    , m_notifierReference(0)
    , m_dispatchQueue(adoptOSObject(dispatch_queue_create("com.apple.WebKit.PowerObserver", 0)))
{
    m_powerConnection = IORegisterForSystemPower(this, &m_notificationPort, [](void* context, io_service_t service, uint32_t messageType, void* messageArgument) {
        static_cast<PowerObserver*>(context)->didReceiveSystemPowerNotification(service, messageType, messageArgument);
    }, &m_notifierReference);
    if (!m_powerConnection)
        return;

    IONotificationPortSetDispatchQueue(m_notificationPort, m_dispatchQueue.get());
}

PowerObserver::~PowerObserver()
{
    if (!m_powerConnection)
        return;

    IODeregisterForSystemPower(&m_notifierReference);
    IOServiceClose(m_powerConnection);
    IONotificationPortDestroy(m_notificationPort);
}

void PowerObserver::didReceiveSystemPowerNotification(io_service_t, uint32_t messageType, void* messageArgument)
{
    IOAllowPowerChange(m_powerConnection, reinterpret_cast<long>(messageArgument));

    // We only care about the "wake from sleep" message.
    if (messageType != kIOMessageSystemWillPowerOn)
        return;

    // We need to restart the timer on the main thread.
    WeakPtr weakThis { *this };
    CFRunLoopPerformBlock(CFRunLoopGetMain(), kCFRunLoopCommonModes, ^() {
        if (weakThis)
            weakThis->m_powerOnHander();
    });
}

} // namespace WebCore

#endif
