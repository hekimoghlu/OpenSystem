/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 27, 2023.
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

#if PLATFORM(IOS_FAMILY)

#if USE(APPLE_INTERNAL_SDK)

#import <IOKit/hid/IOHIDDevice.h>
#import <IOKit/hid/IOHIDEvent.h>
#import <IOKit/hid/IOHIDEventData.h>
#import <IOKit/hid/IOHIDEventSystemClient.h>
#import <IOKit/hid/IOHIDManager.h>
#import <IOKit/hid/IOHIDUsageTables.h>

#else

WTF_EXTERN_C_BEGIN

typedef double IOHIDFloat;

enum {
    kIOHIDEventOptionNone = 0,
};

typedef UInt32 IOOptionBits;
typedef uint32_t IOHIDEventOptionBits;
typedef uint32_t IOHIDEventField;

typedef struct __IOHIDEventSystemClient * IOHIDEventSystemClientRef;
typedef struct __IOHIDEvent * IOHIDEventRef;

#define IOHIDEventFieldBase(type) (type << 16)

enum {
    kHIDPage_KeyboardOrKeypad       = 0x07,
    kHIDPage_VendorDefinedStart     = 0xFF00
};

enum {
    kIOHIDDigitizerEventRange       = 1<<0,
    kIOHIDDigitizerEventTouch       = 1<<1,
    kIOHIDDigitizerEventPosition    = 1<<2,
    kIOHIDDigitizerEventIdentity    = 1<<5,
    kIOHIDDigitizerEventAttribute   = 1<<6,
    kIOHIDDigitizerEventCancel      = 1<<7,
    kIOHIDDigitizerEventStart       = 1<<8,
    kIOHIDDigitizerEventEstimatedAltitude = 1<<28,
    kIOHIDDigitizerEventEstimatedAzimuth = 1<<29,
    kIOHIDDigitizerEventEstimatedPressure = 1<<30
};
typedef uint32_t IOHIDDigitizerEventMask;

enum {
    kIOHIDDigitizerEventUpdateAltitudeMask = 1<<28,
    kIOHIDDigitizerEventUpdateAzimuthMask = 1<<29,
    kIOHIDDigitizerEventUpdatePressureMask = 1<<30
};

enum {
    kIOHIDEventTypeNULL,
    kIOHIDEventTypeVendorDefined,
    kIOHIDEventTypeKeyboard = 3,
    kIOHIDEventTypeRotation = 5,
    kIOHIDEventTypeScroll = 6,
    kIOHIDEventTypeZoom = 8,
    kIOHIDEventTypeDigitizer = 11,
    kIOHIDEventTypeNavigationSwipe = 16,
    kIOHIDEventTypeForce = 32,

};
typedef uint32_t IOHIDEventType;

enum {
    kIOHIDEventFieldVendorDefinedUsagePage = IOHIDEventFieldBase(kIOHIDEventTypeVendorDefined),
    kIOHIDEventFieldVendorDefinedReserved,
    kIOHIDEventFieldVendorDefinedReserved1,
    kIOHIDEventFieldVendorDefinedDataLength,
    kIOHIDEventFieldVendorDefinedData
};

enum {
    kIOHIDEventFieldDigitizerX = IOHIDEventFieldBase(kIOHIDEventTypeDigitizer),
    kIOHIDEventFieldDigitizerY,
    kIOHIDEventFieldDigitizerMajorRadius = kIOHIDEventFieldDigitizerX + 20,
    kIOHIDEventFieldDigitizerMinorRadius,
    kIOHIDEventFieldDigitizerIsDisplayIntegrated = kIOHIDEventFieldDigitizerMajorRadius + 5,
};

enum {
    kIOHIDTransducerRange               = 0x00010000,
    kIOHIDTransducerTouch               = 0x00020000,
    kIOHIDTransducerInvert              = 0x00040000,
    kIOHIDTransducerDisplayIntegrated   = 0x00080000
};

enum {
    kIOHIDDigitizerTransducerTypeStylus  = 0,
    kIOHIDDigitizerTransducerTypeFinger = 2,
    kIOHIDDigitizerTransducerTypeHand = 3
};
typedef uint32_t IOHIDDigitizerTransducerType;

enum {
    kIOHIDEventFieldDigitizerWillUpdateMask = 720924,
    kIOHIDEventFieldDigitizerDidUpdateMask = 720925
};

IOHIDEventRef IOHIDEventCreateDigitizerEvent(CFAllocatorRef, uint64_t, IOHIDDigitizerTransducerType, uint32_t, uint32_t, IOHIDDigitizerEventMask, uint32_t, IOHIDFloat, IOHIDFloat, IOHIDFloat, IOHIDFloat, IOHIDFloat, boolean_t, boolean_t, IOOptionBits);

IOHIDEventRef IOHIDEventCreateDigitizerFingerEvent(CFAllocatorRef, uint64_t, uint32_t, uint32_t, IOHIDDigitizerEventMask, IOHIDFloat, IOHIDFloat, IOHIDFloat, IOHIDFloat, IOHIDFloat, boolean_t, boolean_t, IOHIDEventOptionBits);

IOHIDEventRef IOHIDEventCreateForceEvent(CFAllocatorRef, uint64_t, uint32_t, IOHIDFloat, uint32_t, IOHIDFloat, IOHIDEventOptionBits);

IOHIDEventRef IOHIDEventCreateKeyboardEvent(CFAllocatorRef, uint64_t, uint32_t, uint32_t, boolean_t, IOOptionBits);

IOHIDEventRef IOHIDEventCreateVendorDefinedEvent(CFAllocatorRef, uint64_t, uint32_t, uint32_t, uint32_t, uint8_t*, CFIndex, IOHIDEventOptionBits);

IOHIDEventRef IOHIDEventCreateDigitizerStylusEventWithPolarOrientation(CFAllocatorRef, uint64_t, uint32_t, uint32_t, IOHIDDigitizerEventMask, uint32_t, IOHIDFloat, IOHIDFloat, IOHIDFloat, IOHIDFloat, IOHIDFloat, IOHIDFloat, IOHIDFloat, IOHIDFloat, boolean_t, boolean_t, IOHIDEventOptionBits);

IOHIDEventType IOHIDEventGetType(IOHIDEventRef);

void IOHIDEventSetFloatValue(IOHIDEventRef, IOHIDEventField, IOHIDFloat);

CFIndex IOHIDEventGetIntegerValue(IOHIDEventRef, IOHIDEventField);
void IOHIDEventSetIntegerValue(IOHIDEventRef, IOHIDEventField, CFIndex);

void IOHIDEventAppendEvent(IOHIDEventRef, IOHIDEventRef, IOOptionBits);

IOHIDEventSystemClientRef IOHIDEventSystemClientCreate(CFAllocatorRef);

#define kGSEventPathInfoInRange (1 << 0)
#define kGSEventPathInfoInTouch (1 << 1)

enum {
    kHIDUsage_KeyboardA = 0x04,
    kHIDUsage_Keyboard1 = 0x1E,
    kHIDUsage_Keyboard2 = 0x1F,
    kHIDUsage_Keyboard3 = 0x20,
    kHIDUsage_Keyboard4 = 0x21,
    kHIDUsage_Keyboard5 = 0x22,
    kHIDUsage_Keyboard6 = 0x23,
    kHIDUsage_Keyboard7 = 0x24,
    kHIDUsage_Keyboard8 = 0x25,
    kHIDUsage_Keyboard9 = 0x26,
    kHIDUsage_Keyboard0 = 0x27,
    kHIDUsage_KeyboardReturnOrEnter = 0x28,
    kHIDUsage_KeyboardEscape = 0x29,
    kHIDUsage_KeyboardDeleteOrBackspace = 0x2A,
    kHIDUsage_KeyboardTab = 0x2B,
    kHIDUsage_KeyboardSpacebar = 0x2C,
    kHIDUsage_KeyboardHyphen = 0x2D,
    kHIDUsage_KeyboardEqualSign = 0x2E,
    kHIDUsage_KeyboardOpenBracket = 0x2F,
    kHIDUsage_KeyboardCloseBracket = 0x30,
    kHIDUsage_KeyboardBackslash = 0x31,
    kHIDUsage_KeyboardSemicolon = 0x33,
    kHIDUsage_KeyboardQuote = 0x34,
    kHIDUsage_KeyboardGraveAccentAndTilde = 0x35,
    kHIDUsage_KeyboardComma = 0x36,
    kHIDUsage_KeyboardPeriod = 0x37,
    kHIDUsage_KeyboardSlash = 0x38,
    kHIDUsage_KeyboardCapsLock = 0x39,
    kHIDUsage_KeyboardF1 = 0x3A,
    kHIDUsage_KeyboardF12 = 0x45,
    kHIDUsage_KeyboardPrintScreen = 0x46,
    kHIDUsage_KeyboardInsert = 0x49,
    kHIDUsage_KeyboardHome = 0x4A,
    kHIDUsage_KeyboardPageUp = 0x4B,
    kHIDUsage_KeyboardDeleteForward = 0x4C,
    kHIDUsage_KeyboardEnd = 0x4D,
    kHIDUsage_KeyboardPageDown = 0x4E,
    kHIDUsage_KeyboardRightArrow = 0x4F,
    kHIDUsage_KeyboardLeftArrow = 0x50,
    kHIDUsage_KeyboardDownArrow = 0x51,
    kHIDUsage_KeyboardUpArrow = 0x52,
    kHIDUsage_KeypadNumLock = 0x53,
    kHIDUsage_KeyboardF13 = 0x68,
    kHIDUsage_KeyboardF24 = 0x73,
    kHIDUsage_KeyboardMenu = 0x76,
    kHIDUsage_KeypadComma = 0x85,
    kHIDUsage_KeyboardLeftControl = 0xE0,
    kHIDUsage_KeyboardLeftShift = 0xE1,
    kHIDUsage_KeyboardLeftAlt = 0xE2,
    kHIDUsage_KeyboardLeftGUI = 0xE3,
    kHIDUsage_KeyboardRightControl = 0xE4,
    kHIDUsage_KeyboardRightShift = 0xE5,
    kHIDUsage_KeyboardRightAlt = 0xE6,
    kHIDUsage_KeyboardRightGUI = 0xE7,
};

typedef struct CF_BRIDGED_TYPE(id) __IOHIDDevice * IOHIDDeviceRef;

typedef kern_return_t IOReturn;

enum IOHIDReportType {
    kIOHIDReportTypeInput = 0,
    kIOHIDReportTypeOutput,
};

enum {
    kIOHIDOptionsTypeNone        = 0x00,
    kIOHIDOptionsTypeSeizeDevice = 0x01,
};
typedef uint32_t IOHIDOptionsType;

typedef UInt32 IOOptionBits;

typedef void (*IOHIDReportCallback) (void*, IOReturn, void*, IOHIDReportType, uint32_t, uint8_t*, CFIndex);

IOReturn IOHIDDeviceOpen(IOHIDDeviceRef, IOOptionBits);
void IOHIDDeviceScheduleWithRunLoop(IOHIDDeviceRef, CFRunLoopRef, CFStringRef);
void IOHIDDeviceRegisterInputReportCallback(IOHIDDeviceRef, uint8_t*, CFIndex, IOHIDReportCallback, void*);
void IOHIDDeviceUnscheduleFromRunLoop(IOHIDDeviceRef, CFRunLoopRef, CFStringRef);
IOReturn IOHIDDeviceClose(IOHIDDeviceRef, IOOptionBits);
IOReturn IOHIDDeviceSetReport(IOHIDDeviceRef, IOHIDReportType, CFIndex, const uint8_t*, CFIndex);

typedef struct CF_BRIDGED_TYPE(id) __IOHIDManager * IOHIDManagerRef;

#define kIOHIDPrimaryUsagePageKey "PrimaryUsagePage"
#define kIOHIDPrimaryUsageKey "PrimaryUsage"

typedef void (*IOHIDDeviceCallback) (void*, IOReturn, void*, IOHIDDeviceRef);

IOHIDManagerRef IOHIDManagerCreate(CFAllocatorRef, IOOptionBits);
void IOHIDManagerSetDeviceMatching(IOHIDManagerRef, CFDictionaryRef);
void IOHIDManagerRegisterDeviceMatchingCallback(IOHIDManagerRef, IOHIDDeviceCallback, void*);
void IOHIDManagerRegisterDeviceRemovalCallback(IOHIDManagerRef, IOHIDDeviceCallback, void*);
void IOHIDManagerUnscheduleFromRunLoop(IOHIDManagerRef, CFRunLoopRef, CFStringRef);
IOReturn IOHIDManagerClose(IOHIDManagerRef, IOOptionBits);
void IOHIDManagerScheduleWithRunLoop(IOHIDManagerRef, CFRunLoopRef, CFStringRef);
IOReturn IOHIDManagerOpen(IOHIDManagerRef, IOOptionBits);

WTF_EXTERN_C_END

#endif // USE(APPLE_INTERNAL_SDK)

#endif // PLATFORM(IOS_FAMILY)
