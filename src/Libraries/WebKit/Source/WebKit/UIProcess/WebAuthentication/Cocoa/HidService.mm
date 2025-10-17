/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 24, 2024.
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
#import "HidService.h"

#if ENABLE(WEB_AUTHN)

#import "CtapHidDriver.h"
#import "HidConnection.h"

namespace WebKit {
using namespace fido;

#if HAVE(SECURITY_KEY_API)
// FIXME(191518)
static void deviceAddedCallback(void* context, IOReturn, void*, IOHIDDeviceRef device)
{
    ASSERT(device);
    auto* listener = static_cast<HidService*>(context);
    listener->deviceAdded(device);
}

// FIXME(191518)
static void deviceRemovedCallback(void* context, IOReturn, void*, IOHIDDeviceRef device)
{
    // FIXME(191525)
}
#endif // HAVE(SECURITY_KEY_API)

Ref<HidService> HidService::create(AuthenticatorTransportServiceObserver& observer)
{
    return adoptRef(*new HidService(observer));
}

HidService::HidService(AuthenticatorTransportServiceObserver& observer)
    : FidoService(observer)
{
#if HAVE(SECURITY_KEY_API)
    m_manager = adoptCF(IOHIDManagerCreate(kCFAllocatorDefault, kIOHIDOptionsTypeNone));
    NSDictionary *matchingDictionary = @{
        @kIOHIDPrimaryUsagePageKey: @(kCtapHidUsagePage),
        @kIOHIDPrimaryUsageKey: @(kCtapHidUsage),
    };
    IOHIDManagerSetDeviceMatching(m_manager.get(), (__bridge CFDictionaryRef)matchingDictionary);
    IOHIDManagerRegisterDeviceMatchingCallback(m_manager.get(), deviceAddedCallback, this);
    IOHIDManagerRegisterDeviceRemovalCallback(m_manager.get(), deviceRemovedCallback, this);
#endif
}

HidService::~HidService()
{
#if HAVE(SECURITY_KEY_API)
    IOHIDManagerUnscheduleFromRunLoop(m_manager.get(), CFRunLoopGetCurrent(), kCFRunLoopDefaultMode);
    IOHIDManagerClose(m_manager.get(), kIOHIDOptionsTypeNone);
#endif
}

void HidService::startDiscoveryInternal()
{
    platformStartDiscovery();
}

void HidService::platformStartDiscovery()
{
#if HAVE(SECURITY_KEY_API)
    IOHIDManagerScheduleWithRunLoop(m_manager.get(), CFRunLoopGetCurrent(), kCFRunLoopDefaultMode);
    IOHIDManagerOpen(m_manager.get(), kIOHIDOptionsTypeNone);
#endif
}

Ref<HidConnection> HidService::createHidConnection(IOHIDDeviceRef device) const
{
    return HidConnection::create(device);
}

void HidService::deviceAdded(IOHIDDeviceRef device)
{
#if HAVE(SECURITY_KEY_API)
    getInfo(CtapHidDriver::create(createHidConnection(device)));
#endif
}

} // namespace WebKit

#endif // ENABLE(WEB_AUTHN)
