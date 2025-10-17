/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 11, 2022.
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
//
// SDCSPDLSession.cpp - Security Server CSP/DL session.
//
#include "SDCSPDLSession.h"

#include "SDCSPDLPlugin.h"
#include "SDKey.h"

using namespace SecurityServer;

//
// SDCSPDLSession -- Security Server CSP session
//
SDCSPDLSession::SDCSPDLSession()
{
}


//
// Reference Key management
//
void
SDCSPDLSession::makeReferenceKey(SDCSPSession &session, KeyHandle inKeyHandle,
								 CssmKey &outKey, CSSM_DB_HANDLE inDBHandle,
								 uint32 inKeyAttr, const CssmData *inKeyLabel)
{
	new SDKey(session, inKeyHandle, outKey, inDBHandle, inKeyAttr,
			  inKeyLabel);
}

SDKey &
SDCSPDLSession::lookupKey(const CssmKey &inKey)
{
	/* for now we only allow ref keys */
	if(inKey.blobType() != CSSM_KEYBLOB_REFERENCE) {
		CssmError::throwMe(CSSMERR_CSP_INVALID_KEY);
	}
	
	/* fetch key (this is just mapping the value in inKey.KeyData to an SDKey) */
	SDKey &theKey = find<SDKey>(inKey);
	
	#ifdef someday 
	/* 
	 * Make sure caller hasn't changed any crucial header fields.
	 * Some fields were changed by makeReferenceKey, so make a local copy....
	 */
	CSSM_KEYHEADER localHdr = cssmKey.KeyHeader;
	get binKey-like thing from SDKey, maybe SDKey should keep a copy of 
	hdr...but that's' not supersecure....;
	
	localHdr.BlobType = binKey->mKeyHeader.BlobType;
	localHdr.Format = binKey->mKeyHeader.Format;
	if(memcmp(&localHdr, &binKey->mKeyHeader, sizeof(CSSM_KEYHEADER))) {
		CssmError::throwMe(CSSMERR_CSP_INVALID_KEY_REFERENCE);
	}
	#endif
	return theKey;
}
