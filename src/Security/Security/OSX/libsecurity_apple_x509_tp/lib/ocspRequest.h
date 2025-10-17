/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 22, 2022.
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
 * ocspRequest.h - OCSP Request class
 */

#ifndef	_OCSP_REQUEST_H_
#define _OCSP_REQUEST_H_

#include "TPCertInfo.h"
#include <Security/SecAsn1Coder.h>
#include <Security/ocspTemplates.h>

class OCSPClientCertID;

class OCSPRequest
{
	NOCOPY(OCSPRequest)
public:
	/*
	 * The only constructor. Subject and issuer must remain valid for the
	 * lifetime of this object (they are not refcounted).
	 */
	OCSPRequest(
		TPCertInfo		&subject,
		TPCertInfo		&issuer,
		bool			genNonce);

	~OCSPRequest();

	/*
	 * Obtain encoded OCSP request suitable for posting to responder.
	 * This object owns and maintains the memory.
	 */
	const CSSM_DATA *encode();

	/*
	 * Obtain this request's nonce (which we randomly generate at encode() time),
	 * This object owns and maintains the memory. Result is NULL} if we
	 * didn't generate a nonce.
	 */
	const CSSM_DATA *nonce();

	/*
	 * Obtain this request's CertID. Used to look up matching SingleResponse
	 * in the OCSPResponse.
	 */
	OCSPClientCertID	*certID();

private:
	SecAsn1CoderRef		mCoder;
	TPCertInfo			&mSubject;
	TPCertInfo			&mIssuer;
	bool				mGenNonce;
	CSSM_DATA			mNonce;
	CSSM_DATA			mEncoded;	/* lazily evaluated */
	OCSPClientCertID	*mCertID;	/* calculated during encode() */

};

#endif	/* _OCSP_REQUEST_H_ */

