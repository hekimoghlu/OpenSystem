/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 3, 2024.
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

#if USE(APPLE_INTERNAL_SDK)

#import <GraphicsServices/GraphicsServices.h>

#endif

WTF_EXTERN_C_BEGIN

void GSInitialize(void);
uint64_t GSCurrentEventTimestamp(void);
CFStringRef GSSystemRootDirectory(void);
void GSFontInitialize(void);
void GSFontPurgeFontCache(void);

typedef struct __GSKeyboard* GSKeyboardRef;
uint32_t GSKeyboardGetModifierState(GSKeyboardRef);
Boolean GSEventIsHardwareKeyboardAttached(void);
uint8_t GSEventGetHardwareKeyboardCountry(void);
uint8_t GSEventGetHardwareKeyboardType(void);
void GSEventSetHardwareKeyboardAttached(Boolean attached, uint8_t country);
void GSEventSetHardwareKeyboardAttachedWithCountryCodeAndType(Boolean attached, uint8_t country, uint8_t type);

extern const char *kGSEventHardwareKeyboardAvailabilityChangedNotification;

WTF_EXTERN_C_END

#endif
