/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 9, 2023.
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
// BinaryKey.h - CSP-wide BinaryKey base class
//

#ifndef	_H_BINARY_KEY_
#define _H_BINARY_KEY_

#include <Security/cssmtype.h>
#include <security_cdsa_utilities/cssmkey.h>

// opaque key reference type 
typedef CSSM_INTPTR	KeyRef;

class AppleCSPSession;

/* 
 * unique blob type passed to generateKeyBlob() for key digest calculation 
 */
#define CSSM_KEYBLOB_RAW_FORMAT_DIGEST	\
	(CSSM_KEYBLOB_RAW_FORMAT_VENDOR_DEFINED + 0x12345)


// frame for Binary key; all modules (BSAFE, CryptKit) must subclass
// this and add a member whose type is the native raw key object.
// Subclasses must implement constructor, destructor, and generateKeyBlob().
class BinaryKey
{
public:
						BinaryKey() : mKeyRef(0), mDescData(Allocator::standard()) { }
	virtual 			~BinaryKey() { mKeyRef = 0; }

	/* 
	 * Generate raw key blob.
	 * The format argument is an in/out parameter and is optionally used
	 * to request a specific keyblob format for providers which can generate
	 * multipleÊformats. This value comes from an optional
	 * CSSM_ATTRIBUTE_{PUBLIC,PRIVATE,SYMMETRIC}_KEY_FORMAT attribute in the current
	 * context. If so such attribute is present, the default value 
	 * CSSM_KEYBLOB_RAW_FORMAT_NONE is specified as the default input param.
	 *
	 * All BinaryKeys must handle the special case format 
	 * CSSM_KEYBLOB_RAW_FORMAT_DIGEST, which creates a blob suitable for use
	 * in calcuating the digest of the key blob. 
	 *
	 * The session and paramKey arguments facilitate the conversion of a partial
	 * BinaryKey to a fully formed raw key, i.e., a null wrap to get a fully formed
	 * raw key. The attrFlags aregument is used to indicate that this operation
	 * did in fact convert a partial binary key to a fully formed raw key
	 * (in which case the subclass clears the CSSM_KEYATTR_PARTIAL bit
	 * in attrFlags before returning). 
	 */
	virtual void		generateKeyBlob(
		Allocator 		&allocator,
		CssmData			&blob,
		CSSM_KEYBLOB_FORMAT	&format,	// in/out, CSSM_KEYBLOB_RAW_FORMAT_PKCS1, 
										//   etc.
		AppleCSPSession		&session,
		const CssmKey		*paramKey,	// optional
		CSSM_KEYATTR_FLAGS	&attrFlags)	// IN/OUT

		{
			CssmError::throwMe(CSSMERR_CSP_FUNCTION_NOT_IMPLEMENTED); 
		}

    virtual BinaryKey *getPublicKey() const
    {
        CssmError::throwMe(CSSMERR_CSP_FUNCTION_NOT_IMPLEMENTED);
    }

	CssmKey::Header		mKeyHeader;
	KeyRef				mKeyRef;
	const CssmData		&descData()		{ return mDescData; }
	void				descData(const CssmData &inDescData) 
										{ mDescData.copy(inDescData); }
	
private:
	/* optional DescriptiveData specified by app during WrapKey */
	CssmAutoData		mDescData;
};

// Binary key representing a symmetric key.
class SymmetricBinaryKey : public BinaryKey
{
public:
	SymmetricBinaryKey(
		unsigned keySizeInBits);
	~SymmetricBinaryKey();
	void generateKeyBlob(
		Allocator 		&allocator,
		CssmData			&blob,
		CSSM_KEYBLOB_FORMAT	&format,		/* CSSM_KEYBLOB_RAW_FORMAT_PKCS1, etc. */
		AppleCSPSession		&session,
		const CssmKey		*paramKey,		/* optional, unused here */
		CSSM_KEYATTR_FLAGS 	&attrFlags);	/* IN/OUT */

	CssmData			mKeyData;
	Allocator 		&mAllocator;
};

/*
 * Stateless function to cook up a BinaryKey given a 
 * symmetric CssmKey in RAW format. Returns true on 
 * success, false if we can't deal with this type of key, 
 * throws exception on other runtime errors.
 */
bool symmetricCssmKeyToBinary(
	const CssmKey		&cssmKey,
	BinaryKey			**binKey);	// RETURNED

#endif	// _H_BINARY_KEY_

