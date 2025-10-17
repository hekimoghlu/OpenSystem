/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 8, 2022.
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

#import <wtf/Platform.h>

#if PLATFORM(IOS_FAMILY)

#include <CoreFoundation/CoreFoundation.h>

#if USE(APPLE_INTERNAL_SDK)

#include <MobileGestalt.h>

#else

static const CFStringRef kMGQAppleInternalInstallCapability = CFSTR("apple-internal-install");
static const CFStringRef kMGQMainScreenClass = CFSTR("main-screen-class");
static const CFStringRef kMGQMainScreenPitch = CFSTR("main-screen-pitch");
static const CFStringRef kMGQMainScreenScale = CFSTR("main-screen-scale");
static const CFStringRef kMGQiPadCapability = CFSTR("ipad");
static const CFStringRef kMGQDeviceName = CFSTR("DeviceName");
static const CFStringRef kMGQDeviceClassNumber = CFSTR("DeviceClassNumber");
static const CFStringRef kMGQHasExtendedColorDisplay = CFSTR("HasExtendedColorDisplay");
static const CFStringRef kMGQDeviceCornerRadius = CFSTR("DeviceCornerRadius");
static const CFStringRef kMGQMainScreenStaticInfo CFSTR("MainScreenStaticInfo");
static const CFStringRef kMGQSupportsForceTouch CFSTR("eQd5mlz0BN0amTp/2ccMoA");
static const CFStringRef kMGQBluetoothCapability CFSTR("bluetooth");
static const CFStringRef kMGQDeviceProximityCapability CFSTR("DeviceProximityCapability");
static const CFStringRef kMGQDeviceSupportsARKit CFSTR("arkit");
static const CFStringRef kMGQTimeSyncCapability CFSTR("LJ8aZhTg8lXUeVxHzT+hMw");
static const CFStringRef kMGQWAPICapability CFSTR("wapi");
static const CFStringRef kMGQMainDisplayRotation CFSTR("MainDisplayRotation");

typedef enum {
    MGDeviceClassInvalid       = -1,
    MGDeviceClassiPhone        = 1,
    MGDeviceClassiPod          = 2,
    MGDeviceClassiPad          = 3,
    MGDeviceClassAppleTV       = 4,
    MGDeviceClassWatch         = 6,
    MGDeviceClassMac           = 9,
    MGDeviceClassRealityDevice = 11,
} MGDeviceClass;

typedef enum {
    MGScreenClassPad2          = 4,
    MGScreenClassPad3          = 6,
    MGScreenClassPad4          = 7,
} MGScreenClass;

#endif

#ifdef __OBJC__
@interface MobileGestaltHelperProxy : NSObject
- (BOOL) proxyRebuildCache;
@end
#endif

WTF_EXTERN_C_BEGIN

CFTypeRef MGCopyAnswer(CFStringRef question, CFDictionaryRef options);

#ifndef MGGetBoolAnswer
bool MGGetBoolAnswer(CFStringRef question);
#endif

#ifndef MGGetSInt32Answer
SInt32 MGGetSInt32Answer(CFStringRef question, SInt32 defaultValue);
#endif

#ifndef MGGetFloat32Answer
Float32 MGGetFloat32Answer(CFStringRef question, Float32 defaultValue);
#endif

bool _MGCacheValid();

WTF_EXTERN_C_END

#endif
