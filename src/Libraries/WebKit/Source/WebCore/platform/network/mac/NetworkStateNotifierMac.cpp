/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 23, 2022.
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
#include "config.h"
#include "NetworkStateNotifier.h"

#if PLATFORM(MAC)

#include <SystemConfiguration/SystemConfiguration.h>
#include <wtf/cf/TypeCastsCF.h>

namespace WebCore {

void NetworkStateNotifier::updateStateWithoutNotifying()
{
    if (!m_store)
        return;

    auto key = adoptCF(SCDynamicStoreKeyCreateNetworkInterface(0, kSCDynamicStoreDomainState));
    auto propertyList = dynamic_cf_cast<CFDictionaryRef>(adoptCF(SCDynamicStoreCopyValue(m_store.get(), key.get())));
    if (!propertyList)
        return;

    auto netInterfaces = dynamic_cf_cast<CFArrayRef>(CFDictionaryGetValue(propertyList.get(), kSCDynamicStorePropNetInterfaces));
    if (!netInterfaces)
        return;

    for (CFIndex i = 0; i < CFArrayGetCount(netInterfaces); i++) {
        auto interfaceName = dynamic_cf_cast<CFStringRef>(CFArrayGetValueAtIndex(netInterfaces, i));
        if (!interfaceName)
            continue;

        // Ignore the loopback interface.
        if (CFStringHasPrefix(interfaceName, CFSTR("lo")))
            continue;

        // Ignore Parallels virtual interfaces on host machine as these are always up.
        if (CFStringHasPrefix(interfaceName, CFSTR("vnic")))
            continue;

        // Ignore VMWare virtual interfaces on host machine as these are always up.
        if (CFStringHasPrefix(interfaceName, CFSTR("vmnet")))
            continue;

        auto key = adoptCF(SCDynamicStoreKeyCreateNetworkInterfaceEntity(0, kSCDynamicStoreDomainState, interfaceName, kSCEntNetIPv4));
        if (auto value = adoptCF(SCDynamicStoreCopyValue(m_store.get(), key.get()))) {
            m_isOnLine = true;
            return;
        }
    }

    m_isOnLine = false;
}

void NetworkStateNotifier::startObserving()
{
    SCDynamicStoreContext context = { 0, this, 0, 0, 0 };
    m_store = adoptCF(SCDynamicStoreCreate(0, CFSTR("com.apple.WebCore"), [] (SCDynamicStoreRef, CFArrayRef, void*) {
        // Calling updateState() could be expensive so we coalesce calls with a timer.
        singleton().updateStateSoon();
    }, &context));
    if (!m_store)
        return;

    auto source = adoptCF(SCDynamicStoreCreateRunLoopSource(0, m_store.get(), 0));
    if (!source)
        return;

    CFRunLoopAddSource(CFRunLoopGetMain(), source.get(), kCFRunLoopCommonModes);

    auto keys = adoptCF(CFArrayCreateMutable(0, 0, &kCFTypeArrayCallBacks));
    CFArrayAppendValue(keys.get(), adoptCF(SCDynamicStoreKeyCreateNetworkGlobalEntity(0, kSCDynamicStoreDomainState, kSCEntNetIPv4)).get());
    CFArrayAppendValue(keys.get(), adoptCF(SCDynamicStoreKeyCreateNetworkGlobalEntity(0, kSCDynamicStoreDomainState, kSCEntNetDNS)).get());

    auto patterns = adoptCF(CFArrayCreateMutable(0, 0, &kCFTypeArrayCallBacks));
    CFArrayAppendValue(patterns.get(), adoptCF(SCDynamicStoreKeyCreateNetworkInterfaceEntity(0, kSCDynamicStoreDomainState, kSCCompAnyRegex, kSCEntNetIPv4)).get());

    SCDynamicStoreSetNotificationKeys(m_store.get(), keys.get(), patterns.get());
}
    
}

#endif // PLATFORM(MAC)
