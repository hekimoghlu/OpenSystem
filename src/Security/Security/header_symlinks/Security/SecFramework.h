/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 5, 2024.
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
	@header SecFramework
	The functions provided in SecFramework.h implement generic non API class
    specific functionality.
*/

#ifndef _SECURITY_SECFRAMEWORK_H_
#define _SECURITY_SECFRAMEWORK_H_

#include <CoreFoundation/CFString.h>
#include <CoreFoundation/CFURL.h>
#include <CoreFoundation/CFBundle.h>
#include <Security/SecAsn1Types.h>

__BEGIN_DECLS

#define SecString(key, comment)  CFSTR(key)
#define SecStringFromTable(key, tbl, comment)  CFSTR(key)
#define SecStringWithDefaultValue(key, tbl, bundle, value, comment)  CFSTR(key)

CFBundleRef SecFrameworkGetBundle();

CFStringRef SecFrameworkCopyLocalizedString(CFStringRef key,
    CFStringRef tableName);

Boolean SecFrameworkIsRunningInXcode(void);

/* Return the SHA1 digest of a chunk of data as newly allocated CFDataRef. */
CFDataRef SecSHA1DigestCreate(CFAllocatorRef allocator,
	const UInt8 *data, CFIndex length);

/* Return the SHA256 digest of a chunk of data as newly allocated CFDataRef. */
CFDataRef SecSHA256DigestCreate(CFAllocatorRef allocator,
    const UInt8 *data, CFIndex length);

CFDataRef SecSHA256DigestCreateFromData(CFAllocatorRef allocator, CFDataRef data);

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"

/* Return the digest of a chunk of data as newly allocated CFDataRef, the
   algorithm is selected based on the algorithm and params passed in. */
CFDataRef SecDigestCreate(CFAllocatorRef allocator,
    const SecAsn1Oid *algorithm, const SecAsn1Item *params,
	const UInt8 *data, CFIndex length);

#pragma clang diagnostic pop

// Wrapper to provide a CFErrorRef for legacy API.
OSStatus SecOSStatusWith(bool (^perform)(CFErrorRef *error));

extern CFStringRef kSecFrameworkBundleID;

/* Returns true if 'string' is a DNS host name as defined in RFC 1035, etc. */
bool SecFrameworkIsDNSName(CFStringRef string);

/* Returns true if 'string' is an IPv4/IPv6 address per RFC 2373, 4632, etc. */
bool SecFrameworkIsIPAddress(CFStringRef string);

/* Returns the canonical data representation of the IPv4 or IPv6 address
   provided as input. NULL is returned if string is not a valid IP address. */
CFDataRef SecFrameworkCopyIPAddressData(CFStringRef string);

__END_DECLS

#endif /* !_SECURITY_SECFRAMEWORK_H_ */
