/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 13, 2024.
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
 * FEESignatureObject.cpp - implementations of FEE-style raw sign/verify classes
 *
 */

#ifdef	CRYPTKIT_CSP_ENABLE

#include "FEESignatureObject.h"
#include <security_cryptkit/feePublicKey.h>
#include <security_cryptkit/feeDigitalSignature.h>
#include <security_cryptkit/falloc.h>
#include <stdexcept>
#include <security_utilities/simulatecrash_assert.h>
#include <security_utilities/debugging.h>

#define feeSigObjDebug(args...)		secinfo("feeSig", ##args)

CryptKit::FEESigner::~FEESigner()
{
	if(mWeMallocdFeeKey) {
		assert(mFeeKey != NULL);
		feePubKeyFree(mFeeKey);
	}
}

/*
 * padding from context
 */
void CryptKit::FEESigner::sigFormatFromContext(
                                         const Context 	&context)
{
    CSSM_PADDING padding = context.getInt(CSSM_ATTRIBUTE_PADDING);
    switch(padding) {
        case CSSM_PADDING_SIGRAW:
            mSigFormat=FSF_RAW;
            break;
        default:
            mSigFormat=FSF_DER;
    }
}
/* 
 * obtain key from context, validate, convert to native FEE key
 */
void CryptKit::FEESigner::keyFromContext(
	const Context 	&context)
{
	if(initFlag() && (mFeeKey != NULL)) {
		/* reusing context, OK */
		return;
	}
	
	CSSM_KEYCLASS 	keyClass;
	CSSM_KEYUSE		keyUse;
	if(isSigning()) {
		/* signing with private key */
		keyClass = CSSM_KEYCLASS_PRIVATE_KEY;
		keyUse   = CSSM_KEYUSE_SIGN;
	}
	else {
		/* verifying with public key */
		keyClass = CSSM_KEYCLASS_PUBLIC_KEY;
		keyUse   = CSSM_KEYUSE_VERIFY;
	}
	if(mFeeKey == NULL) {
		mFeeKey = contextToFeeKey(context,
			mSession,
			CSSM_ATTRIBUTE_KEY,
			keyClass,
			keyUse,
			mWeMallocdFeeKey);
	}
}

/* reusable init */
void CryptKit::FEESigner::signerInit(
	const Context 	&context,
	bool			isSigning)
{
	setIsSigning(isSigning);
	keyFromContext(context);
    sigFormatFromContext(context);
	setInitFlag(true);
}

/*
 * Note that, unlike the implementation in security_cryptkit/feePublicKey.c, we ignore
 * the Pm which used to be used as salt for the digest. That made staged verification
 * impossible and I do not believe it increased security. 
 */
void CryptKit::FEERawSigner::sign(
	const void	 	*data, 
	size_t 			dataLen,
	void			*sig,	
	size_t			*sigLen)	/* IN/OUT */
{
	feeSig 			fsig;
	feeReturn		frtn;
	unsigned char	*feeSig = NULL;
	unsigned		feeSigLen=0;
	
	if(mFeeKey == NULL) {
		throwCryptKit(FR_BadPubKey, "FEERawSigner::sign (no key)");
	}
	fsig = feeSigNewWithKey(mFeeKey, mRandFcn, mRandRef);
	if(fsig == NULL) {
		throwCryptKit(FR_BadPubKey, "FEERawSigner::sign");
	}
	frtn = feeSigSign(fsig,
		(unsigned char *)data,
		(unsigned)dataLen,
		mFeeKey);
	if(frtn == FR_Success) {
		frtn = feeSigData(fsig, &feeSig, &feeSigLen);
	}
	feeSigFree(fsig);
	if(frtn) {
		throwCryptKit(frtn, "FEERawSigner::sign");
	}
	
	/* copy out to caller and ffree */
	if(*sigLen < feeSigLen) {
		feeSigObjDebug("FEERawSigner sign overflow\n");
		ffree(feeSig);
		CssmError::throwMe(CSSMERR_CSP_OUTPUT_LENGTH_ERROR);
	}
	memmove(sig, feeSig, feeSigLen);
	*sigLen = feeSigLen;
	ffree(feeSig);
}

void CryptKit::FEERawSigner::verify(
	const void	 	*data, 
	size_t 			dataLen,
	const void		*sig,			
	size_t			sigLen)
{
	feeSig 		fsig;
	feeReturn	frtn;
	
	if(mFeeKey == NULL) {
		throwCryptKit(FR_BadPubKey, "FEERawSigner::verify (no key)");
	}
	frtn = feeSigParse((unsigned char *)sig, sigLen, &fsig);
	if(frtn) {
		throwCryptKit(frtn, "feeSigParse");
	}
	frtn = feeSigVerify(fsig,
		(unsigned char *)data,
		(unsigned int)dataLen,
		mFeeKey);
	feeSigFree(fsig);
	if(frtn) {
		throwCryptKit(frtn, NULL);
	}
}

size_t CryptKit::FEERawSigner::maxSigSize()
{
	unsigned 	rtn;
	feeReturn 	frtn;
	
	frtn = feeSigSize(mFeeKey, &rtn);
	if(frtn) {
		throwCryptKit(frtn, "feeSigSize");
	}
	return rtn;
}

/* ECDSA - this is really easy. */

void CryptKit::FEEECDSASigner::sign(
	const void	 	*data, 
	size_t 			dataLen,
	void			*sig,	
	size_t			*sigLen)	/* IN/OUT */
{
	unsigned char	*feeSig;
	unsigned		feeSigLen;
	feeReturn		frtn;
	
	if(mFeeKey == NULL) {
		throwCryptKit(FR_BadPubKey, "FEERawSigner::sign (no key)");
	}
	frtn = feeECDSASign(mFeeKey,
        mSigFormat,
		(unsigned char *)data,   // data to be signed
		(unsigned int)dataLen,				// in bytes
		mRandFcn, 
		mRandRef,
		&feeSig,
		&feeSigLen);			
	if(frtn) {
		throwCryptKit(frtn, "feeECDSASign");
	}
	/* copy out to caller and ffree */
	if(*sigLen < feeSigLen) {
		feeSigObjDebug("feeECDSASign overflow\n");
		ffree(feeSig);
		CssmError::throwMe(CSSMERR_CSP_OUTPUT_LENGTH_ERROR);
	}
	memmove(sig, feeSig, feeSigLen);
	*sigLen = feeSigLen;
	ffree(feeSig);

}

void CryptKit::FEEECDSASigner::verify(
	const void	*data, 
	size_t 		dataLen,
	const void	*sig,			
	size_t		sigLen)
{
	feeReturn	frtn;

	if(mFeeKey == NULL) {
		throwCryptKit(FR_BadPubKey, "FEERawSigner::verify (no key)");
	}
	frtn = feeECDSAVerify(
        (unsigned char *)sig,
		sigLen,
		(unsigned char *)data,
		(unsigned int)dataLen,
		mFeeKey,
        mSigFormat);
	if(frtn) {
		throwCryptKit(frtn, NULL);
	}
}

size_t CryptKit::FEEECDSASigner::maxSigSize()
{
	unsigned 	rtn;
	feeReturn 	frtn;
	
	frtn = feeECDSASigSize(mFeeKey, &rtn);
	if(frtn) {
		throwCryptKit(frtn, "feeECDSASigSize");
	}
	return rtn;
}

#endif	/* CRYPTKIT_CSP_ENABLE */
