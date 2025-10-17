/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 30, 2022.
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
// cryptoclient - client interface to CSSM CSP encryption/decryption operations
//
#include <security_cdsa_client/cryptoclient.h>

using namespace CssmClient;


Crypt::Crypt(const CSP &csp, CSSM_ALGORITHMS alg)
	: Context(csp, alg), mMode(CSSM_ALGMODE_NONE), mInitVector(NULL),
	  mPadding(CSSM_PADDING_NONE)
{
}

void Crypt::key(const Key &key)
{
	mKey = key;
	set(CSSM_ATTRIBUTE_KEY, static_cast<const CssmKey &>(key));
}


void
Crypt::activate()
{
    StLock<Mutex> _(mActivateMutex);
	if (!mActive)
	{
        // Key is required unless we have a NULL algorithm (cleartext wrap/unwrap),
        // in which case we'll make a symmetric context (it shouldn't matter then).
		if (!mKey && mAlgorithm != CSSM_ALGID_NONE)
			CssmError::throwMe(CSSMERR_CSP_MISSING_ATTR_KEY);
		if (!mKey || mKey->keyClass() == CSSM_KEYCLASS_SESSION_KEY)
		{	// symmetric key
			check(CSSM_CSP_CreateSymmetricContext(attachment()->handle(), mAlgorithm,
				mMode, neededCred(), mKey, mInitVector, mPadding, NULL,
				&mHandle));
		}
		else
		{
			check(CSSM_CSP_CreateAsymmetricContext(attachment()->handle(), mAlgorithm,
				neededCred(), mKey, mPadding, &mHandle));
			//@@@ stick mode and initVector explicitly into the context?
		}		
		mActive = true;
	}
}


//
// Manage encryption contexts
//
CSSM_SIZE
Encrypt::encrypt(const CssmData *in, uint32 inCount,
						CssmData *out, uint32 outCount, CssmData &remData)
{
	unstaged();
	CSSM_SIZE total;
	check(CSSM_EncryptData(handle(), in, inCount, out, outCount, &total, &remData));
	return total;
}

void
Encrypt::init()
{
	check(CSSM_EncryptDataInit(handle()));
	mStaged = true;
}

CSSM_SIZE
Encrypt::encrypt(const CssmData *in, uint32 inCount,
	CssmData *out, uint32 outCount)
{
	staged();
	CSSM_SIZE total;
	check(CSSM_EncryptDataUpdate(handle(), in, inCount, out, outCount, &total));
	return total;
}

void
Encrypt::final(CssmData &remData)
{
	staged();
	check(CSSM_EncryptDataFinal(handle(), &remData));
	mStaged = false;
}


//
// Manage Decryption contexts
//

CSSM_SIZE
Decrypt::decrypt(const CssmData *in, uint32 inCount,
	CssmData *out, uint32 outCount, CssmData &remData)
{
    // So we can free any memory the underlying layers may have allocated before
    // throwing, that way all such callers need not free in the error case.
    // Allocation can happen if either `*out` or `remData` wraps NULL.
    // The underlying Writer will only allocate for the first CssmData in the
    // array, so we only have to worry about that one.
    // Note: A lower layer will throw if `out` is NULL, but we don't want to crash here.
    bool mightAllocateOut = out != NULL && out->data() == NULL;
    bool mightAllocateRem = remData.data() == NULL;

	unstaged();
	CSSM_SIZE total;
    try {
        check(CSSM_DecryptData(handle(), in, inCount, out, outCount, &total, &remData));
    } catch (...) {
        if (mightAllocateOut && out->data() != NULL) {
            Allocator::standard().free(out->data());
            out->clear();
        }
        if (mightAllocateRem && remData.data() != NULL) {
            Allocator::standard().free(remData.data());
            remData.clear();
        }
        throw;
    }
	return total;
}

void
Decrypt::init()
{
	check(CSSM_DecryptDataInit(handle()));
	mStaged = true;
}

CSSM_SIZE
Decrypt::decrypt(const CssmData *in, uint32 inCount,
	CssmData *out, uint32 outCount)
{
	staged();
	CSSM_SIZE total;
	check(CSSM_DecryptDataUpdate(handle(), in, inCount, out, outCount, &total));
	return total;
}

void
Decrypt::final(CssmData &remData)
{
	staged();
	check(CSSM_DecryptDataFinal(handle(), &remData));
	mStaged = false;
}
