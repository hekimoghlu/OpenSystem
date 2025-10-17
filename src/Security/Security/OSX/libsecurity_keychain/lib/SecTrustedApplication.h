/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 17, 2025.
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
	@header SecTrustedApplication
	The functions provided in SecTrustedApplication implement an object representing an application in a
	SecAccess object.
*/

#ifndef _SECURITY_SECTRUSTEDAPPLICATION_H_
#define _SECURITY_SECTRUSTEDAPPLICATION_H_

#include <Security/SecBase.h>
#include <CoreFoundation/CoreFoundation.h>


#if defined(__cplusplus)
extern "C" {
#endif

CF_ASSUME_NONNULL_BEGIN

/*!
	@function SecTrustedApplicationGetTypeID
	@abstract Returns the type identifier of SecTrustedApplication instances.
	@result The CFTypeID of SecTrustedApplication instances.
*/
CFTypeID SecTrustedApplicationGetTypeID(void)
API_DEPRECATED("SecKeychain is deprecated", macos(10.2, 10.10))
API_UNAVAILABLE(ios, watchos, tvos, bridgeos, macCatalyst);

/*!
	@function SecTrustedApplicationCreateFromPath
    @abstract Creates a trusted application reference based on the trusted application specified by path.
    @param path The path to the application or tool to trust. For application bundles, use the
		path to the bundle directory. Pass NULL to refer to yourself, i.e. the application or tool
		making this call.
    @param app On return, a pointer to the trusted application reference.
    @result A result code.  See "Security Error Codes" (SecBase.h).
*/
OSStatus SecTrustedApplicationCreateFromPath(const char * __nullable path, SecTrustedApplicationRef * __nonnull CF_RETURNS_RETAINED app)
API_DEPRECATED("SecKeychain is deprecated", macos(10.0, 10.10))
API_UNAVAILABLE(ios, watchos, tvos, bridgeos, macCatalyst);

/*!
	@function SecTrustedApplicationCopyData
	@abstract Retrieves the data of a given trusted application reference
	@param appRef A trusted application reference to retrieve data from
	@param data On return, a pointer to a data reference of the trusted application.
	@result A result code.  See "Security Error Codes" (SecBase.h).
*/
OSStatus SecTrustedApplicationCopyData(SecTrustedApplicationRef appRef, CFDataRef * __nonnull CF_RETURNS_RETAINED data)
API_DEPRECATED("SecKeychain is deprecated", macos(10.0, 10.10))
API_UNAVAILABLE(ios, watchos, tvos, bridgeos, macCatalyst);

/*!
	@function SecTrustedApplicationSetData
	@abstract Sets the data of a given trusted application reference
	@param appRef A trusted application reference.
	@param data A reference to the data to set in the trusted application.
	@result A result code.  See "Security Error Codes" (SecBase.h).
*/
OSStatus SecTrustedApplicationSetData(SecTrustedApplicationRef appRef, CFDataRef data)
API_DEPRECATED("SecKeychain is deprecated", macos(10.0, 10.10))
API_UNAVAILABLE(ios, watchos, tvos, bridgeos, macCatalyst);

CF_ASSUME_NONNULL_END

#if defined(__cplusplus)
}
#endif

#endif /* !_SECURITY_SECTRUSTEDAPPLICATION_H_ */
