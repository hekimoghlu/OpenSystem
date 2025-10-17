/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 17, 2023.
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
 * tpOcspCache.h - local OCSP response cache.
 */
 
#ifndef	_TP_OCSP_CACHE_H_
#define _TP_OCSP_CACHE_H_

#include <security_ocspd/ocspResponse.h>

/* max default TTL currently 12 hours */
#define TP_OCSP_CACHE_TTL	(60.0 * 60.0 * 12.0)

extern "C" {

/*
 * Lookup locally cached response. Caller must free the returned OCSPSingleResponse.
 * Never returns a stale entry; we always check the enclosed SingleResponse for
 * temporal validity.
 */
OCSPSingleResponse *tpOcspCacheLookup(
	OCSPClientCertID	&certID,
	const CSSM_DATA		*localResponderURI);		// optional 

/* 
 * Add a fully verified OCSP response to cache. 
 */
void tpOcspCacheAdd(
	const CSSM_DATA		&ocspResp,				// we'll decode it and own the result
	const CSSM_DATA		*localResponderURI);	// optional 

/*
 * Delete any entry associated with specified certID from cache.
 */
void tpOcspCacheFlush(
	OCSPClientCertID	&certID);

}
#endif	/* _TP_OCSP_CACHE_H_ */

