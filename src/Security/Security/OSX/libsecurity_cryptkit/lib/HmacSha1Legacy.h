/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 1, 2022.
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
 	File:		HmacSha1Legacy.h
 	Contains:	HMAC/SHA1, bug-for-bug compatible a legacy implementation.
 	Copyright (c) 2001,2011-2014 Apple Inc. All Rights Reserved.
*/
#ifndef __HMAC_SHA1_LEGACY__
#define __HMAC_SHA1_LEGACY__

#if	!defined(__MACH__)
#include <ckconfig.h>
#else
#include <security_cryptkit/ckconfig.h>
#endif

#include <MacTypes.h>

#ifdef	__cplusplus
extern "C" {
#endif

/*
 * This version is bug-for-bug compatible with the HMACSHA1 implementation in 
 * an old crypto library. 
 */
struct hmacLegacyContext;
typedef struct hmacLegacyContext *hmacLegacyContextRef;

hmacLegacyContextRef hmacLegacyAlloc(void);
void hmacLegacyFree(
	hmacLegacyContextRef hmac);
OSStatus hmacLegacyInit(
	hmacLegacyContextRef hmac,
	const void *keyPtr,
	UInt32 keyLen);
OSStatus hmacLegacyUpdate(
	hmacLegacyContextRef hmac,
	const void *textPtr,
	UInt32 textLen);
OSStatus hmacLegacyFinal(
	hmacLegacyContextRef hmac,
	void *resultPtr);		// caller mallocs, must be kSHA1DigestSize bytes

#ifdef	__cplusplus
}
#endif

#endif	/* __HMAC_SHA1_LEGACY__ */
