/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 20, 2024.
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
#import <HID/HIDServiceClient.h>
#import <IOKit/hid/IOHIDServiceClient.h>
#import <os/assumes.h>

@implementation HIDServiceClient (HIDFramework)

- (id)propertyForKey:(NSString *)key
{
    return (id)CFBridgingRelease(IOHIDServiceClientCopyProperty(
                                        (__bridge IOHIDServiceClientRef)self,
                                        (__bridge CFStringRef)key));
}

- (NSDictionary *)propertiesForKeys:(NSArray<NSString *> *)keys
{
    return (NSDictionary *)CFBridgingRelease(IOHIDServiceClientCopyProperties(
                                        (__bridge IOHIDServiceClientRef)self,
                                        (__bridge CFArrayRef)keys));
}

- (BOOL)setProperty:(id)value forKey:(NSString *)key
{
    return IOHIDServiceClientSetProperty((__bridge IOHIDServiceClientRef)self,
                                         (__bridge CFStringRef)key,
                                         (__bridge CFTypeRef)value);
}

- (BOOL)conformsToUsagePage:(NSInteger)usagePage usage:(NSInteger)usage
{
    return IOHIDServiceClientConformsTo((__bridge IOHIDServiceClientRef)self,
                                        (uint32_t)usagePage,
                                        (uint32_t)usage);
}

- (HIDEvent *)eventMatching:(NSDictionary *)matching
{
    return (HIDEvent *)CFBridgingRelease(IOHIDServiceClientCopyMatchingEvent(
                                        (__bridge IOHIDServiceClientRef)self,
                                        (__bridge CFDictionaryRef)matching));
}

static void _removalCallback(void *target __unused,
                             void *refcon __unused,
                             IOHIDServiceClientRef service)
{
    HIDServiceClient *me = (__bridge HIDServiceClient *)service;
    
    if (me->_client.removalHandler) {
        ((__bridge HIDBlock)me->_client.removalHandler)();
        Block_release(me->_client.removalHandler);
        me->_client.removalHandler = nil;
    }
}

- (void)setRemovalHandler:(HIDBlock)handler
{
    os_unfair_recursive_lock_lock(&_client.callbackLock);
    os_assert(!_client.removalHandler, "Removal handler already set");
    _client.removalHandler = (void *)Block_copy((__bridge const void *)handler);
    os_unfair_recursive_lock_unlock(&_client.callbackLock);
    IOHIDServiceClientRegisterRemovalCallback(
                                        (__bridge IOHIDServiceClientRef)self,
                                        _removalCallback,
                                        nil,
                                        nil);
}

- (uint64_t)serviceID
{
    id regID = (__bridge id)IOHIDServiceClientGetRegistryID(
                                        (__bridge IOHIDServiceClientRef)self);
    return regID ? [regID unsignedLongLongValue] : 0;
}

@end
