/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 10, 2025.
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
#ifndef _H_SECCODEHOST
#define _H_SECCODEHOST

#include <Security/CSCommon.h>

#ifdef __cplusplus
extern "C" {
#endif

CF_ASSUME_NONNULL_BEGIN

CF_ENUM(uint32_t) {
	kSecCSDedicatedHost = 1 << 0,
	kSecCSGenerateGuestHash = 1 << 1,
};

OSStatus SecHostCreateGuest(SecGuestRef host,
	uint32_t status, CFURLRef path, CFDictionaryRef __nullable attributes,
	SecCSFlags flags, SecGuestRef * __nonnull newGuest)
AVAILABLE_MAC_OS_X_VERSION_10_5_AND_LATER_BUT_DEPRECATED_IN_MAC_OS_X_VERSION_10_6;
OSStatus SecHostRemoveGuest(SecGuestRef host, SecGuestRef guest, SecCSFlags flags)
AVAILABLE_MAC_OS_X_VERSION_10_5_AND_LATER_BUT_DEPRECATED_IN_MAC_OS_X_VERSION_10_6;
OSStatus SecHostSelectGuest(SecGuestRef guestRef, SecCSFlags flags)
AVAILABLE_MAC_OS_X_VERSION_10_5_AND_LATER_BUT_DEPRECATED_IN_MAC_OS_X_VERSION_10_6;
OSStatus SecHostSelectedGuest(SecCSFlags flags, SecGuestRef * __nonnull guestRef)
AVAILABLE_MAC_OS_X_VERSION_10_5_AND_LATER_BUT_DEPRECATED_IN_MAC_OS_X_VERSION_10_6;
OSStatus SecHostSetGuestStatus(SecGuestRef guestRef,
	uint32_t status, CFDictionaryRef __nullable attributes,
	SecCSFlags flags)
	AVAILABLE_MAC_OS_X_VERSION_10_5_AND_LATER_BUT_DEPRECATED_IN_MAC_OS_X_VERSION_10_6;
OSStatus SecHostSetHostingPort(mach_port_t hostingPort, SecCSFlags flags)
AVAILABLE_MAC_OS_X_VERSION_10_5_AND_LATER_BUT_DEPRECATED_IN_MAC_OS_X_VERSION_10_6;

CF_ASSUME_NONNULL_END

#ifdef __cplusplus
}
#endif

#endif //_H_SECCODEHOST
