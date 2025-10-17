/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 11, 2022.
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

#if USE(APPLE_INTERNAL_SDK)

#import <CoreFoundation/CFPriv.h>

#else

#include <CoreFoundation/CoreFoundation.h>

typedef CF_ENUM(CFIndex, CFSystemVersion) {
    CFSystemVersionLion = 7,
    CFSystemVersionMountainLion = 8,
};

typedef CF_OPTIONS(uint64_t, __CFRunLoopOptions) {
    __CFRunLoopOptionsEnableAppNap = 0x3b000000
};

#endif

WTF_EXTERN_C_BEGIN

extern const CFStringRef kCFWebServicesProviderDefaultDisplayNameKey;
extern const CFStringRef kCFWebServicesTypeWebSearch;
extern const CFStringRef _kCFSystemVersionBuildVersionKey;
extern const CFStringRef _kCFSystemVersionProductUserVisibleVersionKey;
extern const CFStringRef _kCFSystemVersionProductVersionKey;

Boolean _CFAppVersionCheckLessThan(CFStringRef bundleID, int linkedOnAnOlderSystemThan, double versionNumberLessThan);
Boolean _CFExecutableLinkedOnOrAfter(CFSystemVersion);
CFDictionaryRef _CFCopySystemVersionDictionary();
CFDictionaryRef _CFWebServicesCopyProviderInfo(CFStringRef serviceType, Boolean* outIsUserSelection);

void __CFRunLoopSetOptionsReason(__CFRunLoopOptions opts, CFStringRef reason);

#ifdef __OBJC__
void _CFPrefsSetDirectModeEnabled(BOOL enabled);
#endif
void _CFPrefsSetReadOnly(Boolean flag);

WTF_EXTERN_C_END
