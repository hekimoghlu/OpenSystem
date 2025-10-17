/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 18, 2023.
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
 * DecodedCrl.cpp - object representing a decoded CRL, in NSS format,
 * with extensions parsed and decoded (still in NSS format).
 *
 */

#include "DecodedCrl.h"
#include "cldebugging.h"
#include "AppleX509CLSession.h"
#include "CSPAttacher.h"
#include <Security/cssmapple.h>

DecodedCrl::DecodedCrl(
	AppleX509CLSession	&session)
	: DecodedItem(session)
{
	memset(&mCrl, 0, sizeof(mCrl));
}

/* one-shot constructor, decoding from DER-encoded data */
DecodedCrl::DecodedCrl(
	AppleX509CLSession	&session,
	const CssmData 		&encodedCrl)
	: DecodedItem(session)
{
	memset(&mCrl, 0, sizeof(mCrl));
	PRErrorCode prtn = mCoder.decode(encodedCrl.data(), encodedCrl.length(), 
		kSecAsn1SignedCrlTemplate, &mCrl);
	if(prtn) {
		CssmError::throwMe(CSSMERR_CL_UNKNOWN_FORMAT);
	}
	mDecodedExtensions.decodeFromNss(mCrl.tbs.extensions);
	mState = IS_DecodedAll;
}
		
DecodedCrl::~DecodedCrl()
{
}
	
/* decode mCrl.tbs and its extensions */
void DecodedCrl::decodeCts(
	const CssmData	&encodedCts)
{
	assert(mState == IS_Empty);
	memset(&mCrl, 0, sizeof(mCrl));
	PRErrorCode prtn = mCoder.decode(encodedCts.data(), encodedCts.length(), 
		kSecAsn1TBSCrlTemplate, &mCrl.tbs);
	if(prtn) {
		CssmError::throwMe(CSSMERR_CL_UNKNOWN_FORMAT);
	}
	mDecodedExtensions.decodeFromNss(mCrl.tbs.extensions);
	mState = IS_DecodedTBS;
}

void DecodedCrl::encodeExtensions()
{
	NSS_TBSCrl &tbs = mCrl.tbs;
	assert(mState == IS_Building);
	assert(tbs.extensions == NULL);

	if(mDecodedExtensions.numExtensions() == 0) {
		/* no extensions, no error */
		return;
	}
	mDecodedExtensions.encodeToNss(tbs.extensions);
}

/*
 * FIXME : how to determine max encoding size at run time!?
 */
#define MAX_TEMPLATE_SIZE	(16 * 1024)

/* encode TBS component; only called from CrlCreateTemplate */
void DecodedCrl::encodeCts(
	CssmOwnedData	&encodedCts)
{
	encodeExtensions();
	assert(mState == IS_Building);
	
	/* enforce required fields - could go deeper, maybe we should */
	NSS_TBSCrl &tbs = mCrl.tbs;
	if((tbs.signature.algorithm.Data == NULL) ||
	   (tbs.issuer.rdns == NULL)) {
		clErrorLog("DecodedCrl::encodeTbs: incomplete TBS");
		/* an odd, undocumented error return */
		CssmError::throwMe(CSSMERR_CL_NO_FIELD_VALUES);
	}
	
	PRErrorCode prtn;
	prtn = SecNssEncodeItemOdata(&tbs, kSecAsn1TBSCrlTemplate,
		encodedCts);
	if(prtn) {
		CssmError::throwMe(CSSMERR_CL_MEMORY_ERROR);
	}
}

