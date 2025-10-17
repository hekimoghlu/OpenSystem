/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 26, 2025.
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
/*
 	File:		HMACSHA1.h
 	Contains:	Apple Data Security Services HMAC{SHA1,MD5} function declaration.
 	Copyright (c) 1999,2011,2013-2014 Apple Inc. All Rights Reserved.
*/
#ifndef __HMACSHA1__
#define __HMACSHA1__

#include <Security/cssmtype.h>
#include <pbkdDigest.h>
#include <CommonCrypto/CommonDigest.h>

#ifdef	__cplusplus
extern "C" {
#endif

#define kHMACSHA1DigestSize  	CC_SHA1_DIGEST_LENGTH
#define kHMACMD5DigestSize	 	CC_MD5_DIGEST_LENGTH

/* This function create an HMACSHA1 digest of kHMACSHA1DigestSizestSize bytes
 * and outputs it to resultPtr.  See RFC 2104 for details.  */
void
hmacsha1 (const void *keyPtr, uint32 keyLen,
		  const void *textPtr, uint32 textLen,
		  void *resultPtr);
		  
#ifdef	__cplusplus
}
#endif

#endif /* __HMACSHA1__ */
