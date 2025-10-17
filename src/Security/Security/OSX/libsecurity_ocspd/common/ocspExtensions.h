/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 10, 2021.
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
 * ocspExtensions.h - OCSP extensions support
 */
 
#ifndef	_OCSP_EXTENSIONS_H_
#define _OCSP_EXTENSIONS_H_

#include <Security/SecAsn1Coder.h>
#include <Security/x509defs.h>
#include <Security/X509Templates.h>
#include <security_utilities/utilities.h>

/* 
 * We deal with a well bounded set of extensions, so we can enumerate them
 * here for convenience. 
 */
typedef enum {
	OET_Unknown,			// no recognized
	OET_Nonce,
	OET_CrlReference,
	OET_AcceptResponse,
	OET_ArchiveCutoff,
	OET_ServiceLocator
} OCSPExtensionTag;

class OCSPExtension 
{
	NOCOPY(OCSPExtension);
	/* note NO public constructor implemented by this class */
public:
	/* the public means to create an OCSPExtension subclass during decode */
	static OCSPExtension *createFromNSS(
		SecAsn1CoderRef			coder,
		const NSS_CertExtension &nssExt);
	
	virtual ~OCSPExtension();
	
	/* public accessors; suclass probably has others */
	bool critical()					{ return mCritical; }
	bool unrecognizedCritical()		{ return mUnrecognizedCritical; }
	CSSM_OID &extnId()				{ return mNssExt->extnId; }
	OCSPExtensionTag tag()			{ return mTag; }
	
	/* 
	 * When encoding, this is ready to go - i.e., we're ready to be encoded - 
	 * once subclass has called setDerValue(). That happens during subclass's
	 * constructor.
	 */
	NSS_CertExtension *nssExt() { return mNssExt; }
	
protected:
	/* 
	 * Subclass must implement a version like this (without the tag argument), 
	 * called from createFromNSS() during decode.
	 *
	 * This class's implementation just stashes away mNssExt, mCritical, and mCoder.
	 * This class's implementation is also used to construct the "I don't understand
	 * this extension" case (tag = OET_Unknown). 
	 */
	OCSPExtension(
		SecAsn1CoderRef			coder,
		const NSS_CertExtension	&nssExt,
		OCSPExtensionTag		tag);
		
	/* 
	 * Constructor during encode, called from subclass-specific constructorÊ(which
	 * always has all of the subclass-specific arguments).
	 */ 
	OCSPExtension(
		SecAsn1CoderRef			coder,			// passed to subclass constructor
		const CSSM_OID			&extnId,		// subclass knows this 
		OCSPExtensionTag		tag,			// subclass knows this
		bool					critical);		// passed to subclass constructor
		
	/*
	 * Called by subclass after it DER encodes its NSS_CertExtension.value. The
	 * data is always in our coder's address space (so we don't copy it here).
	 */
	void setDerValue(
		const CSSM_DATA			&derValue)		{ mNssExt->value = (CSSM_DATA)derValue; }
		
private:
	/* 
	 * This NSS_CertExtension is always in mCoder's address space. 
	 * -- On decode, createFromNSS()'s caller provides a reference to it
	 *    after high-level decode of a SecAsn1OCSPSignedRequest or a
	 *    SecAsn1OCSPResponseData. 
	 * -- On encode, *we* (this class) allocate with our mCoder.
	 */
	NSS_CertExtension		*mNssExt;
    [[maybe_unused]] SecAsn1CoderRef			mCoder;
	bool					mCritical;			// convenience - it's also in mNssExt
	OCSPExtensionTag		mTag;
	
	/* set true when mCritical for extension we don't understand */
	bool					mUnrecognizedCritical;
};

class OCSPNonce : public OCSPExtension
{
	NOCOPY(OCSPNonce);
public:
	/* construct during encode, called directly by app */
	OCSPNonce(
		SecAsn1CoderRef		coder,
		bool				critical,
		const CSSM_DATA		&nonce);

	virtual ~OCSPNonce();
	
	CSSM_DATA &nonce()		{ return mNonce; }
	
protected:
	friend class OCSPExtension;
	/* construct during decode, called only by OCSPExtension::createFromNSS() */
	OCSPNonce(
		SecAsn1CoderRef			coder,
		const NSS_CertExtension &nssExt);

private:
		CSSM_DATA			mNonce;
};


/*
 * A simple class representing an array of extensions, used during decoding.
 */
class OCSPExtensions
{
	NOCOPY(OCSPExtensions);
public:
	/* 
	 * Create from array of NSS_CertExtensions, which is the form available 
	 * immediately after decoding. We'll throw if we find any decoding error,
	 * or if we find a critical extension this module does not understand.
	 */
	OCSPExtensions(
		NSS_CertExtension **nssExts);
	~OCSPExtensions();
	
	/* 
	 * Find a decoded extension associated with specified extnId. Returns
	 * NULL if not found.
	 * Caller will typically cast (dynamic_cast would be appropriate) to 
	 * a specific extension type.
	 */
	OCSPExtension *findExtension(
		const CSSM_OID		&oid);
		
private:
	SecAsn1CoderRef				mCoder;
	unsigned					mNumExtensions;
	OCSPExtension				**mExtensions;
};

#endif  /* _OCSP_EXTENSIONS_H_ */
