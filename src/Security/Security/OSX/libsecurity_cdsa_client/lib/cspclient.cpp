/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 29, 2024.
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
// cspclient - client interface to CSSM CSPs and their operations
//
#include <security_cdsa_client/cspclient.h>

namespace Security {
namespace CssmClient {


//
// Manage CSP attachments
//
CSPImpl::CSPImpl(const Guid &guid) : AttachmentImpl(guid, CSSM_SERVICE_CSP)
{
}

CSPImpl::CSPImpl(const Module &module) : AttachmentImpl(module, CSSM_SERVICE_CSP)
{
}

CSPImpl::~CSPImpl()
{
}


//
// Delete a key explicitly
//
void CSPImpl::freeKey(CssmKey &key, const AccessCredentials *cred, bool permanent)
{
    check(CSSM_FreeKey(handle(), cred, &key, permanent));
}


//
// Manage generic context objects
//
Context::Context(const CSP &csp, CSSM_ALGORITHMS alg) : ObjectImpl(csp), mAlgorithm(alg), mHandle(NULL), mStaged(false), mCred(NULL)
{
}

Context::~Context()
{
	try
	{
		deactivate();
	} catch(...) {}
}

void Context::init()
{
	CssmError::throwMe(CSSM_ERRCODE_FUNCTION_NOT_IMPLEMENTED);
}

void Context::deactivate()
{
    StLock<Mutex> _(mActivateMutex);
	if (mActive)
	{
		mActive = false;
		check(CSSM_DeleteContext(mHandle));
	}
}


void Context::algorithm(CSSM_ALGORITHMS alg)
{
	if (isActive())
		abort();	//@@@ can't (currently?) change algorithm with active context
	mAlgorithm = alg;
}


void Context::cred(const CSSM_ACCESS_CREDENTIALS *cred)
{
	mCred = AccessCredentials::overlay(cred);
    set(CSSM_ATTRIBUTE_ACCESS_CREDENTIALS, *mCred);
}


//
// Query context operation output sizes.
//    
uint32 Context::getOutputSize(uint32 inputSize, bool encrypt /*= true*/)
{
    CSSM_QUERY_SIZE_DATA data;
    data.SizeInputBlock = inputSize;
    getOutputSize(data, 1, encrypt);
    return data.SizeOutputBlock;
}

void Context::getOutputSize(CSSM_QUERY_SIZE_DATA &sizes, uint32 count, bool encrypt /*= true*/)
{
    check(CSSM_QuerySize(handle(), encrypt, count, &sizes));
}


//
// The override() method of Context is an expert feature. It replaces the entire
// context with a context object provided. It is up to the caller to keep this context
// consistent with the purpose of the Context subclass he is (mis)using.
// This feature is currently used by the SecurityServer.
//
void Context::override(const Security::Context &ctx)
{
	if (!isActive()) {
		// make a valid context object (it doesn't matter what kind - keep it cheap)
		check(CSSM_CSP_CreateDigestContext(attachment()->handle(), CSSM_ALGID_NONE, &mHandle));
	}
	// now replace everything with the context data provided
	check(CSSM_SetContext(mHandle, &ctx));
	mActive = true;		// now active
}


//
// RccContexts
//
const ResourceControlContext &RccBearer::compositeRcc() const
{
	// explicitly specified RCC wins
	if (mRcc)
		return *mRcc;
	
	// cobble one up from the pieces
	if (mOwner)
		mWorkRcc.input() = *mOwner;
	else
		mWorkRcc.clearPod();
	mWorkRcc.credentials(mOpCred);
	return mWorkRcc;
}


void RccBearer::owner(const CSSM_ACL_ENTRY_PROTOTYPE *owner)
{
	if (owner) {
		mWorkInput = *owner;
		this->owner(mWorkInput);
	} else
		this->owner((AclEntryInput*)NULL);
}


//
// Manage PassThrough contexts
//

//
// Invoke passThrough
//
void
PassThrough::operator() (uint32 passThroughId, const void *inData, void **outData)
{
    check(CSSM_CSP_PassThrough(handle(), passThroughId, inData, outData));
}

void PassThrough::activate()
{
    StLock<Mutex> _(mActivateMutex);
	if (!mActive) {
		check(CSSM_CSP_CreatePassThroughContext(attachment()->handle(), mKey, &mHandle));
		mActive = true;
	}
}


//
// Manage Digest contexts
//
void Digest::activate()
{
    StLock<Mutex> _(mActivateMutex);
	if (!mActive) {
		check(CSSM_CSP_CreateDigestContext(attachment()->handle(), mAlgorithm, &mHandle));
		mActive = true;
	}
}


void Digest::digest(const CssmData *data, uint32 count, CssmData &digest)
{
	activate();
	if (mStaged)
		Error::throwMe(CSSMERR_CSP_STAGED_OPERATION_IN_PROGRESS);
	check(CSSM_DigestData(handle(), data, count, &digest));
}

void Digest::digest(const CssmData *data, uint32 count)
{
	activate();
	if (!mStaged) {
		check(CSSM_DigestDataInit(handle()));
		mStaged = true;
	}
	check(CSSM_DigestDataUpdate(handle(), data, count));
}

void Digest::operator () (CssmData &digest)
{
	if (!mStaged)
		Error::throwMe(CSSMERR_CSP_STAGED_OPERATION_NOT_STARTED);
	check(CSSM_DigestDataFinal(handle(), &digest));
	mStaged = false;
}


//
// Random number generation
//
void Random::seed(const CssmCryptoData &seedData)
{
	mSeed = &seedData;
	set(CSSM_ATTRIBUTE_SEED, seedData);
}

void Random::size(uint32 sz)
{
	mSize = sz;
	set(CSSM_ATTRIBUTE_OUTPUT_SIZE, sz);
}


void Random::activate()
{
    StLock<Mutex> _(mActivateMutex);
	if (!mActive) {
		check(CSSM_CSP_CreateRandomGenContext(attachment()->handle(), mAlgorithm,
			mSeed, mSize, &mHandle));
		mActive = true;
	}
}


void Random::generate(CssmData &data, uint32 newSize)
{
	if (newSize)
		size(newSize);
	activate();
	assert(!mStaged);	// not a stage-able operation
	check(CSSM_GenerateRandom(handle(), &data));
}

} // end namespace CssmClient
} // end namespace Security
