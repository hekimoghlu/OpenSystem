/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 7, 2021.
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

#include <CoreFoundation/CoreFoundation.h>

#if PLATFORM(MAC) || USE(APPLE_INTERNAL_SDK)

#include <IOKit/pwr_mgt/IOPMLib.h>

#else

#include <wtf/spi/cocoa/IOReturnSPI.h>

typedef uint32_t IOPMAssertionID;

WTF_EXTERN_C_BEGIN

const CFStringRef kIOPMAssertionTypePreventUserIdleDisplaySleep = CFSTR("PreventUserIdleDisplaySleep");
const CFStringRef kIOPMAssertionTypePreventUserIdleSystemSleep = CFSTR("PreventUserIdleSystemSleep");

WTF_EXTERN_C_END

#endif

WTF_EXTERN_C_BEGIN

IOReturn IOPMAssertionCreateWithDescription(CFStringRef assertionType, CFStringRef name, CFStringRef details, CFStringRef humanReadableReason,
    CFStringRef localizationBundlePath, CFTimeInterval timeout, CFStringRef timeoutAction, IOPMAssertionID *);
IOReturn IOPMAssertionRelease(IOPMAssertionID);

WTF_EXTERN_C_END
