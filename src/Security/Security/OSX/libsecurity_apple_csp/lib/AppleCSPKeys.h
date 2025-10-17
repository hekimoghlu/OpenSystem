/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 22, 2025.
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
 * AppleCSPKeys.h - Key support
 */
 
#ifndef	_APPLE_CSP_KEYS_H_
#define _APPLE_CSP_KEYS_H_

#include "AppleCSPSession.h"

/*
 * Class to provide key-specific info. Each module dealing with keys
 * implements one of these. It's sort of like a CSP-specific CSPContext
 * without the Context object. AppleCSPSession finds one of these by
 * querying module-specific subclasses, looking for one in which 
 * the constructor succeeds (which occurs when the specified key 
 * meets a subclass's specification). 
 */
class CSPKeyInfoProvider 
{
protected:
	CSPKeyInfoProvider(
		const CssmKey 	&cssmKey,
		AppleCSPSession &session) : 
			mKey(cssmKey),
			mSession(session) { }
public:
	/* 
	 * This is the public way to construct - returns NULL if key is 
	 * not handled. Static declaration per subclass.
	 *
	 * static CSPKeyInfoProvider *provider(
	 *	const CssmKey  &cssmKey,
	 * Â AppleCSPSession	&session);
	 */	 
	virtual ~CSPKeyInfoProvider() { }
	
	/* 
	 * Cook up a Binary key. 
	 *
	 * Incoming paramKey optionally contains a key from which 
	 * additional algorithm parameters may be obtained to create
	 * a fully specified key in case the key provided to our
	 * constructor was a partial key. 
	 *
	 * The attrFlags argument is a means for the info provider to 
	 * inform the caller that the incoming key has additional
	 * attributes, currently CSSM_KEYATTR_PARTIAL. The provider
	 * ORs in bits as appropriate. 
	 */
	virtual void CssmKeyToBinary(
		CssmKey				*paramKey,		// optional
		CSSM_KEYATTR_FLAGS	&attrFlags,		// IN/OUT
		BinaryKey			**binKey) = 0;	// RETURNED
		
	/* obtain key size in bits */
	virtual void QueryKeySizeInBits(
		CSSM_KEY_SIZE		&keySize) = 0;	// RETURNED
		
	/* 
	 * Get blob appropriate for calculating key digest, if possible
	 * to do without generating a BinaryKey. Returns true if
	 * successful, falseif BinaryKey generation is required.
	 */
	virtual bool getHashableBlob(
		Allocator 	&allocator,
		CssmData		&hashBlob) = 0;	// blob to hash goes here

protected:
	const CssmKey			&mKey;
	AppleCSPSession			&mSession;
};

/*
 * CSPKeyInfoProvider for symmetric keys (handled directly by
 * the session). 
 */
class SymmetricKeyInfoProvider : public CSPKeyInfoProvider 
{
private:
	SymmetricKeyInfoProvider(
		const CssmKey		&cssmKey,
		AppleCSPSession		&session);
public:
	static CSPKeyInfoProvider *provider(
		const CssmKey 		&cssmKey,
		AppleCSPSession		&session);
		
	~SymmetricKeyInfoProvider() { }
	void CssmKeyToBinary(
		CssmKey				*paramKey,	// ignored
		CSSM_KEYATTR_FLAGS	&attrFlags,	// IN/OUT
		BinaryKey			**binKey);	// RETURNED
	void QueryKeySizeInBits(
		CSSM_KEY_SIZE		&keySize);	// RETURNED
	bool getHashableBlob(
		Allocator 		&allocator,
		CssmData			&hashBlob);
};

#endif	/* _APPLE_CSP_KEYS_H_ */

