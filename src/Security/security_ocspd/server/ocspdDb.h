/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 28, 2022.
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
 * ocspdDb.h - API for OCSP daemon database
 */
 
#ifndef	_OCSPD_DB_H_
#define _OCSPD_DB_H_

#include <Security/cssmtype.h>
#include <Security/SecAsn1Coder.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Lookup cached response. Result is a DER-encoded OCSP response, the same bits
 * originally obtained from the net. Result is allocated in specified 
 * SecAsn1CoderRef's memory. Never returns a stale entry; we always check the 
 * enclosed SingleResponse for temporal validity. 
 *
 * Just a boolean returned; we found it, or not.
 */
bool ocspdDbCacheLookup(
	SecAsn1CoderRef		coder,
	const CSSM_DATA		&certID,
	const CSSM_DATA		*localResponder,	// optional; if present, must match
											// entry's URI
	CSSM_DATA			&derResp);			// RETURNED

/* 
 * Add an OCSP response to cache. Incoming response is completely unverified;
 * we just verify that we can parse it and is has at least one SingleResponse
 * which is temporally valid. 
 */
void ocspdDbCacheAdd(
	const CSSM_DATA		&ocspResp,			// as it came from the server
	const CSSM_DATA		&URI);				// where it came from 

/*
 * Delete any entry associated with specified certID from cache.
 */
void ocspdDbCacheFlush(
	const CSSM_DATA		&certID);

/*
 * Flush stale entries from cache. 
 */
void ocspdDbCacheFlushStale();

#ifdef __cplusplus
}
#endif

#endif	/* _OCSPD_DB_H_ */

