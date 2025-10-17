/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 17, 2021.
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
 * ocspExtensions.cpp - OCSP Extension support.  
 */

#include "ocspExtensions.h"
#include "ocspdDebug.h"
#include "ocspdUtils.h"
#include <Security/oidscrl.h>
#include <Security/cssmapple.h>
#include <strings.h>
#include "ocspdDebug.h"
#include <security_cdsa_utilities/cssmerrors.h>

#pragma mark ----- base class : OCSCExtension -----

/* 
 * Public means to vend a subclass of this object while decoding. 
 */
OCSPExtension *OCSPExtension::createFromNSS(
	SecAsn1CoderRef		coder,
	const NSS_CertExtension	&nssExt)
{
	const CSSM_OID *extnId = &nssExt.extnId;
	if(ocspdCompareCssmData(extnId, &CSSMOID_PKIX_OCSP_NONCE)) {
		return new OCSPNonce(coder, nssExt);
	}
	/* more here */
	else {
		return new OCSPExtension(coder, nssExt, OET_Unknown);
	}
}

/* 
 * Called in two circumstances:
 * 
 * -- from subclass-specific constructor during decode
 * -- from createFromNSS (during decode) when we don't recognize the extension ID
 */
OCSPExtension::OCSPExtension(
	SecAsn1CoderRef			coder,
	const NSS_CertExtension	&nssExt,
	OCSPExtensionTag		tag)
		: mNssExt(const_cast<NSS_CertExtension *>(&nssExt)), 
		  mCoder(coder), 
		  mTag(tag),
		  mUnrecognizedCritical(false)
{
	if((nssExt.critical.Data != NULL) && (*nssExt.critical.Data != 0)) {
		mCritical = true;
	}
	else {
		mCritical = false;
	}
	if(mCritical && (tag == OET_Unknown)) {
		mUnrecognizedCritical = true;
	}
}

/* 
 * Constructor during encode, called from subclass-specific constructorÊ(which
 * always has all of the subclass-specific arguments).
 */ 
OCSPExtension::OCSPExtension(
	SecAsn1CoderRef			coder,			// passed to subclass constructor
	const CSSM_OID			&extnId,		// subclass knows this 
	OCSPExtensionTag		tag,			// subclass knows this
	bool					critical)		// passed to subclass constructor
		: mNssExt(NULL),					// we'll cook this up
		  mCoder(coder),
		  mCritical(critical),
		  mTag(tag),
		  mUnrecognizedCritical(false)		// this is a tautology
{
	mNssExt = (NSS_CertExtension *)SecAsn1Malloc(coder, sizeof(NSS_CertExtension));
	memset(mNssExt, 0, sizeof(NSS_CertExtension));
	SecAsn1AllocCopyItem(coder, &extnId, &mNssExt->extnId);
	/* alloc one byte for critical flag */
	SecAsn1AllocItem(coder, &mNssExt->critical, 1);
	mNssExt->critical.Data[0] = critical ? 0xff : 0;
}

OCSPExtension::~OCSPExtension()
{
	/* nothing for now, need a virtual function for dynamic casts */
}

#pragma mark ---- Nonce -----

/* 
 * Public constructor on encode
 */
OCSPNonce::OCSPNonce(
	SecAsn1CoderRef		coder,
	bool				critical,
	const CSSM_DATA		&nonce)
		: OCSPExtension(coder, CSSMOID_PKIX_OCSP_NONCE, OET_Nonce, critical)
{
	/* 
	 * They don't get much simpler than this: the nonce is the literal value
	 * of NSS_CertExtension.value.
	 */
	SecAsn1AllocCopyItem(coder, &nonce, &mNonce);
	setDerValue(mNonce);
}

/* construct during decode, called only by OCSPExtension::createFromNSS() */
OCSPNonce::OCSPNonce(
	SecAsn1CoderRef		coder,
	const NSS_CertExtension &nssExt)
		: OCSPExtension(coder, nssExt, OET_Nonce)
{
	/* only subclass-specific stuff is the nonce, no further processing needed */
	SecAsn1AllocCopyItem(coder, &nssExt.value, &mNonce);
}

OCSPNonce::~OCSPNonce()
{
	/* nothing for now, need a virtual function for dynamic casts */
}

#pragma mark ----- Extensions array -----

OCSPExtensions::OCSPExtensions(
	NSS_CertExtension **nssExts)
		: mCoder(NULL), mNumExtensions(0), mExtensions(NULL)
{
	SecAsn1CoderCreate(&mCoder);
	mNumExtensions = ocspdArraySize((const void **)nssExts);
	if(mNumExtensions == 0) {
		return;
	}
	
	mExtensions = (OCSPExtension **)SecAsn1Malloc(mCoder, 
		(mNumExtensions * sizeof(OCSPExtension *)));
	for(unsigned dex=0; dex<mNumExtensions; dex++) {
		try {
			mExtensions[dex] = 
				OCSPExtension::createFromNSS(mCoder, *nssExts[dex]);
			if(mExtensions[dex] == NULL) {
				ocspdErrorLog("OCSPExtensions: extension failure (NULL) dex %u\n", dex);
				CssmError::throwMe(CSSMERR_APPLETP_OCSP_BAD_RESPONSE);
			}
			if(mExtensions[dex]->unrecognizedCritical()) {
				ocspdErrorLog("OCSPExtensions: unrecognized critical extension\n");
				CssmError::throwMe(CSSMERR_APPLETP_OCSP_BAD_RESPONSE);
			}
		}
		catch (...) {
			ocspdErrorLog("OCSPExtensions: extension failure dex %u\n", dex);
			CssmError::throwMe(CSSMERR_APPLETP_OCSP_BAD_RESPONSE);
		}
	}
}

OCSPExtensions::~OCSPExtensions()
{
	for(unsigned dex=0; dex<mNumExtensions; dex++) {
		delete mExtensions[dex];
	}
	if(mCoder) {
		SecAsn1CoderRelease(mCoder);
	}
}

/* find parsed extension in mExtensions with specified OID */
OCSPExtension *OCSPExtensions::findExtension(
	const CSSM_OID &oid)
{
	for(unsigned dex=0; dex<mNumExtensions; dex++) {
		const CSSM_OID &extnId = mExtensions[dex]->extnId();
		if(ocspdCompareCssmData(&oid, &extnId)) {
			return mExtensions[dex];
		}
	}
	return NULL;
}

