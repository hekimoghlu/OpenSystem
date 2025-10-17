/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 14, 2024.
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
 * DecodedItem.cpp - class representing the common portions of NSS-style
 * certs and CRLs, with extensions parsed and decoded (still in NSS
 * format).
 */

#include "DecodedItem.h"
#include "cldebugging.h"
#include "AppleX509CLSession.h"
#include "CSPAttacher.h"
#include "CLFieldsCommon.h"
#include "clNssUtils.h"
#include <Security/cssmapple.h>


DecodedItem::DecodedItem(
	AppleX509CLSession	&session)
	:	mState(IS_Empty),
		mAlloc(session),
		mSession(session),
		mDecodedExtensions(mCoder, session)
{
}

DecodedItem::~DecodedItem()
{
	/* nothing for now */
}

/* 
 * Search for DecodedExten by AsnOid or "any unknown extension".
 * Called from getField*() and inferKeyUsage. 
 * Returns NULL if specified extension not found.
 */
const DecodedExten *DecodedItem::findDecodedExt(
	const CSSM_OID		&extnId,		// for known extensions
	bool				unknown,		// otherwise		
	uint32				index, 
	uint32				&numFields) const
{
	unsigned dex;
	const DecodedExten *rtnExt = NULL;
	unsigned found = 0;
	
	for(dex=0; dex<mDecodedExtensions.numExtensions(); dex++) {
		const DecodedExten *decodedExt = mDecodedExtensions.getExtension(dex);
		/*
		 * known extensions: OID match AND successful decode (In case
		 *    we encountered a known extension which we couldn't
		 *    decode and fell back to giving the app an unparsed
		 *    BER blob). 
		 * unknown extensions: just know that we didn't decode it
		 */
		if( ( !unknown && !decodedExt->berEncoded() &&
		      (clCompareCssmData(&decodedExt->extnId(), &extnId))
			) || 
		    (unknown && decodedExt->berEncoded())
		   ) {
			
			if(found++ == index) {
				/* the one we want */
				rtnExt = decodedExt;
			}
			if((rtnExt != NULL) && (index != 0)) {
				/* only determine numFields on search for first one */
				break;
			}
		}
	}
	if(rtnExt != NULL) {
		/* successful return  */
		if(index == 0) {
			numFields = found;
		}
		return rtnExt;
	}
	else {
		return NULL;
	}
}

