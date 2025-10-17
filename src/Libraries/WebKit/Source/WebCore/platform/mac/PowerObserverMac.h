/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 1, 2024.
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
#ifndef PowerObserverMac_h
#define PowerObserverMac_h

#import <IOKit/IOMessage.h>
#import <IOKit/pwr_mgt/IOPMLib.h>
#import <wtf/Function.h>
#import <wtf/Noncopyable.h>
#import <wtf/OSObjectPtr.h>
#import <wtf/TZoneMalloc.h>
#import <wtf/WeakPtr.h>

namespace WebCore {
class PowerObserver;
}

namespace WTF {
template<typename T> struct IsDeprecatedWeakRefSmartPointerException;
template<> struct IsDeprecatedWeakRefSmartPointerException<WebCore::PowerObserver> : std::true_type { };
}

namespace WebCore {

class PowerObserver : public CanMakeWeakPtr<PowerObserver, WeakPtrFactoryInitialization::Eager> {
    WTF_MAKE_TZONE_ALLOCATED_EXPORT(PowerObserver, WEBCORE_EXPORT);
    WTF_MAKE_NONCOPYABLE(PowerObserver);

public:
    WEBCORE_EXPORT PowerObserver(Function<void()>&& powerOnHander);
    WEBCORE_EXPORT ~PowerObserver();

private:
    void didReceiveSystemPowerNotification(io_service_t, uint32_t messageType, void* messageArgument);

    Function<void()> m_powerOnHander;
    io_connect_t m_powerConnection;
    IONotificationPortRef m_notificationPort;
    io_object_t m_notifierReference;
    OSObjectPtr<dispatch_queue_t> m_dispatchQueue;
};

} // namespace WebCore

#endif // PowerObserverMac_h

