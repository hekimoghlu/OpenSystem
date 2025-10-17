/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 1, 2023.
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
 * AppleCSPKeys.cpp - Key support
 */
 
#include "AppleCSPKeys.h"
#include "AppleCSPUtils.h"
/*
 * CSPKeyInfoProvider for symmetric keys. 
 */
CSPKeyInfoProvider *SymmetricKeyInfoProvider::provider(
		const CssmKey 	&cssmKey,
		AppleCSPSession	&session)
{
	if(cssmKey.blobType() != CSSM_KEYBLOB_RAW) {
		errorLog0("KeyInfoProvider deals only with RAW keys!\n");
		CssmError::throwMe(CSSMERR_CSP_INTERNAL_ERROR);
	}
	if(cssmKey.keyClass() != CSSM_KEYCLASS_SESSION_KEY) {
		/* that's all we need to know */
		return NULL;
	}
	return new SymmetricKeyInfoProvider(cssmKey, session);
}
 
SymmetricKeyInfoProvider::SymmetricKeyInfoProvider(
	const CssmKey 	&cssmKey,
	AppleCSPSession	&session) :
		CSPKeyInfoProvider(cssmKey, session)
{
}

/* cook up a Binary key */
void SymmetricKeyInfoProvider::CssmKeyToBinary(
	CssmKey				*paramKey,	// ignored
	CSSM_KEYATTR_FLAGS	&attrFlags,	// IN/OUT
	BinaryKey 			**binKey)
{
	CASSERT(mKey.keyClass() == CSSM_KEYCLASS_SESSION_KEY);
	SymmetricBinaryKey *symBinKey = new SymmetricBinaryKey(
		mKey.KeyHeader.LogicalKeySizeInBits);
	copyCssmData(mKey, 
		symBinKey->mKeyData, 
		symBinKey->mAllocator);
	*binKey = symBinKey;
}

/* obtain key size in bits */
void SymmetricKeyInfoProvider::QueryKeySizeInBits(
	CSSM_KEY_SIZE &keySize)
{
	/* FIXME - do we ever need to calculate RC2 effective size here? */
	keySize.LogicalKeySizeInBits = keySize.EffectiveKeySizeInBits =
		(uint32)(mKey.length() * 8);
}

/* 
 * Obtain blob suitable for hashing in CSSM_APPLECSP_KEYDIGEST 
 * passthrough.
 */
bool SymmetricKeyInfoProvider::getHashableBlob(
	Allocator 	&allocator,
	CssmData		&blob)			// blob to hash goes here
{
	/*
	 * This is trivial: the raw key is already in the "proper" format.
	 */
	assert(mKey.blobType() == CSSM_KEYBLOB_RAW);
	const CssmData &keyBlob = CssmData::overlay(mKey.KeyData);
	copyCssmData(keyBlob, blob, allocator);
	return true;
}

