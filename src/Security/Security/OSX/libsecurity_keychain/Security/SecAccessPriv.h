/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 1, 2024.
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
/*!
	@header SecAccessPriv
	SecAccessPriv implements a way to set and manipulate access control rules and
	restrictions on SecKeychainItems. The functions here are private.
*/

#ifndef _SECURITY_SECACCESS_PRIV_H_
#define _SECURITY_SECACCESS_PRIV_H_

#include <Security/SecBase.h>
#include <Security/cssmtype.h>
#include <CoreFoundation/CFArray.h>


#if defined(__cplusplus)
extern "C" {
#endif

/*!
	@function SecAccessCreateWithTrustedApplications
	@abstract Creates a SecAccess object with the specified trusted applications.
    @param trustedApplicationsPListPath A full path to the .plist file that contains the trusted applications. The extension must end in ".plist".
	@param accessLabel The access label for the new SecAccessRef.
	@param allowAny Flag that determines allow access to any application.
	@param returnedAccess On return, a new SecAccessRef.
	@result A result code.  See "Security Error Codes" (SecBase.h).
	@discussion The SecAccessCreateWithPList creates a SecAccess with the provided list of trusted applications.
*/

OSStatus SecAccessCreateWithTrustedApplications(CFStringRef trustedApplicationsPListPath, CFStringRef accessLabel, Boolean allowAny, SecAccessRef* returnedAccess)
API_DEPRECATED("SecKeychain is deprecated", macos(10.2, 10.10))
API_UNAVAILABLE(ios, watchos, tvos, bridgeos, macCatalyst);


#if defined(__cplusplus)
}
#endif

#endif /* !_SECURITY_SECACCESS_PRIV_H_ */
