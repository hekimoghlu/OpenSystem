/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 27, 2024.
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
 * FEEKeys.h - FEE-related asymmetric key pair classes. 
 *
 */

#ifdef	CRYPTKIT_CSP_ENABLE

#ifndef	_FEE_KEYS_H_
#define _FEE_KEYS_H_

#include "AppleCSPContext.h"
#include "AppleCSPSession.h"
#include "AppleCSPKeys.h"
#include "cryptkitcsp.h"
#include <security_cryptkit/feeTypes.h>

namespace CryptKit {

/*
 * FEE/ECDSA version of a BinaryKey.
 */
class FEEBinaryKey : public BinaryKey {
public:
	FEEBinaryKey(feePubKey feeKey = NULL);
	~FEEBinaryKey();
	void generateKeyBlob(
		Allocator 		&allocator,
		CssmData			&blob,
		CSSM_KEYBLOB_FORMAT	&format,
		AppleCSPSession		&session,
		const CssmKey		*paramKey,		/* optional, unused here */
		CSSM_KEYATTR_FLAGS 	&attrFlags);	/* IN/OUT */
    BinaryKey *getPublicKey() const;

	feePubKey				feeKey() const { return mFeeKey; }
private:
	feePubKey				mFeeKey;
};

class FEEKeyPairGenContext : 
	public AppleCSPContext, private AppleKeyPairGenContext  {
public:
	FEEKeyPairGenContext(
		AppleCSPSession &session,
		const Context &) :
			AppleCSPContext(session) {}

	~FEEKeyPairGenContext() { }
	
	/* no init functionality, but we need to implement it */
	void init(
		const Context &, 
		bool) { }
		
	// this one is specified in, and called from, CSPFullPluginSession
	void generate(
		const Context 	&context, 
		CssmKey 		&pubKey, 
		CssmKey 		&privKey);

    // declared in CSPFullPluginSession, but not implemented here
    void generate(const Context &context, uint32, CssmData &params, uint32 &attrCount, Context::Attr * &attrs);
		
	// this one is specified in, and called from, AppleKeyPairGenContext
	void generate(
		const Context 	&context,
		BinaryKey		&pubBinKey,	
		BinaryKey		&privBinKey,
		uint32			&keySize);
	
};	/* KeyPairGenContext */

/*
 * CSPKeyInfoProvider for FEE and ECDSA keys
 */
class FEEKeyInfoProvider : public CSPKeyInfoProvider 
{
private:
	FEEKeyInfoProvider(
		const CssmKey		&cssmKey,
		AppleCSPSession		&session);
public:
	static CSPKeyInfoProvider *provider(
		const CssmKey 		&cssmKey,
		AppleCSPSession		&session);
		
	~FEEKeyInfoProvider() { }
	void CssmKeyToBinary(
		CssmKey				*paramKey,	// optional, ignored here
		CSSM_KEYATTR_FLAGS	&attrFlags,	// IN/OUT
		BinaryKey			**binKey);	// RETURNED
	void QueryKeySizeInBits(
		CSSM_KEY_SIZE		&keySize);	// RETURNED
	bool getHashableBlob(
		Allocator 		&allocator,
		CssmData			&hashBlob);
};

} /* namespace CryptKit */

#endif	/* _FEE_KEYS_H_ */
#endif	/* CRYPTKIT_CSP_ENABLE */
