/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 15, 2023.
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
 * crlDb.h - CRL cache
 */
 
#ifndef	_OCSPD_CRL_DB_H_
#define _OCSPD_CRL_DB_H_

#include <Security/cssmtype.h>
#include <security_utilities/alloc.h>
#include <security_utilities/debugging.h>


#ifdef __cplusplus
extern "C" {
#endif

/*
 * Lookup cached CRL by URL or issuer, and verifyTime. 
 * Just a boolean returned; we found it, or not.
 * Exactly one of {url, issuer} should be non-NULL.
 */
bool crlCacheLookup(
	Allocator			&alloc,
	const CSSM_DATA		*url,
	const CSSM_DATA		*issuer,			// optional
	const CSSM_DATA		&verifyTime,
	CSSM_DATA			&crlData);			// allocd in alloc space and RETURNED

/* 
 * Add a CRL response to cache. Incoming response is completely unverified;
 * we just verify that we can parse it. 
 */
CSSM_RETURN crlCacheAdd(
	const CSSM_DATA		&crlData,			// as it came from the server
	const CSSM_DATA		&url);				// where it came from 

/*
 * Delete any CRL associated with specified URL from cache.
 */
void crlCacheFlush(
	const CSSM_DATA		&url);

/* 
 * Refresh the CRL cache. 
 */
void crlCacheRefresh(
	unsigned			staleDays,
	unsigned			expireOverlapSeconds,
	bool				purgeAll,
	bool				fullCryptoVerify,
	bool				doRefresh);

#ifdef __cplusplus
}
#endif

#endif	/* _OCSPD_CRL_DB_H_ */

