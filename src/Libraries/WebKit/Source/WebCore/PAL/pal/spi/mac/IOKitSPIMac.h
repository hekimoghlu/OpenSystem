/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 4, 2023.
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

#if PLATFORM(MAC)

#import <IOKit/hid/IOHIDDevice.h>
#import <IOKit/hid/IOHIDManager.h>
#import <IOKit/hid/IOHIDUsageTables.h>

#if USE(APPLE_INTERNAL_SDK)

#import <IOKit/hid/IOHIDEvent.h>
#import <IOKit/hid/IOHIDEventData.h>
#import <IOKit/hid/IOHIDEventSystemClient.h>

#else

#define kIOHIDVendorIDKey "VendorID"
#define kIOHIDProductIDKey "ProductID"

WTF_EXTERN_C_BEGIN

typedef struct __IOHIDEvent * IOHIDEventRef;
typedef struct CF_BRIDGED_TYPE(id) __IOHIDServiceClient * IOHIDServiceClientRef;
typedef struct CF_BRIDGED_TYPE(id) __IOHIDEventSystemClient * IOHIDEventSystemClientRef;
typedef void (^IOHIDServiceClientBlock)(void *, void *, IOHIDServiceClientRef);

typedef CF_ENUM(int, IOHIDEventSystemClientType)
{
    kIOHIDEventSystemClientTypeAdmin,
    kIOHIDEventSystemClientTypeMonitor,
    kIOHIDEventSystemClientTypePassive,
    kIOHIDEventSystemClientTypeRateControlled,
    kIOHIDEventSystemClientTypeSimple
};

IOHIDEventSystemClientRef IOHIDEventSystemClientCreateWithType(CFAllocatorRef, IOHIDEventSystemClientType, CFDictionaryRef);
IOHIDEventSystemClientRef IOHIDEventSystemClientCreate(CFAllocatorRef);
void IOHIDEventSystemClientSetMatching(IOHIDEventSystemClientRef, CFDictionaryRef);
void IOHIDEventSystemClientSetMatchingMultiple(IOHIDEventSystemClientRef, CFArrayRef);
IOHIDServiceClientRef IOHIDEventSystemClientCopyServiceForRegistryID(IOHIDEventSystemClientRef, uint64_t registryID);
void IOHIDEventSystemClientRegisterDeviceMatchingBlock(IOHIDEventSystemClientRef, IOHIDServiceClientBlock, void *, void *);
void IOHIDEventSystemClientUnregisterDeviceMatchingBlock(IOHIDEventSystemClientRef);
void IOHIDEventSystemClientScheduleWithDispatchQueue(IOHIDEventSystemClientRef, dispatch_queue_t);
void IOHIDEventSystemClientSetDispatchQueue(IOHIDEventSystemClientRef, dispatch_queue_t);
void IOHIDEventSystemClientActivate(IOHIDEventSystemClientRef);

CFTypeRef IOHIDServiceClientCopyProperty(IOHIDServiceClientRef service, CFStringRef key);

enum {
    kIOHIDEventTypeNULL,
    kIOHIDEventTypeVendorDefined,
    kIOHIDEventTypeKeyboard = 3,
    kIOHIDEventTypeRotation = 5,
    kIOHIDEventTypeScroll = 6,
    kIOHIDEventTypeZoom = 8,
    kIOHIDEventTypeDigitizer = 11,
    kIOHIDEventTypeNavigationSwipe = 16,
    kIOHIDEventTypeZoomToggle = 22,
    kIOHIDEventTypeForce = 32,
};
typedef uint32_t IOHIDEventType;

typedef uint32_t IOHIDEventField;
typedef uint64_t IOHIDEventSenderID;


enum {
    kIOHIDEventScrollMomentumInterrupted = (1 << 4),
};
typedef uint8_t IOHIDEventScrollMomentumBits;

#ifdef __LP64__
typedef double IOHIDFloat;
#else
typedef float IOHIDFloat;
#endif

#define IOHIDEventFieldBase(type) (type << 16)

#define kIOHIDEventFieldScrollBase IOHIDEventFieldBase(kIOHIDEventTypeScroll)
static const IOHIDEventField kIOHIDEventFieldScrollX = (kIOHIDEventFieldScrollBase | 0);
static const IOHIDEventField kIOHIDEventFieldScrollY = (kIOHIDEventFieldScrollBase | 1);

uint64_t IOHIDEventGetTimeStamp(IOHIDEventRef);
IOHIDFloat IOHIDEventGetFloatValue(IOHIDEventRef, IOHIDEventField);
IOHIDEventSenderID IOHIDEventGetSenderID(IOHIDEventRef);
IOHIDEventScrollMomentumBits IOHIDEventGetScrollMomentum(IOHIDEventRef);

WTF_EXTERN_C_END

#endif // USE(APPLE_INTERNAL_SDK)
#endif // PLATFORM(MAC)
