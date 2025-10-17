/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 23, 2025.
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
// signclient - client interface to CSSM sign/verify contexts
//
#include <security_cdsa_client/signclient.h>

using namespace CssmClient;


//
// Common features of signing and verify contexts
//
void SigningContext::activate()
{
    StLock<Mutex> _(mActivateMutex);
	if (!mActive)
	{
		check(CSSM_CSP_CreateSignatureContext(attachment()->handle(), mAlgorithm,
			  cred(), mKey, &mHandle));
		mActive = true;
	}
}


//
// Signing
//
void Sign::sign(const CssmData *data, uint32 count, CssmData &signature)
{
	unstaged();
	check(CSSM_SignData(handle(), data, count, mSignOnly, &signature));
}

void Sign::init()
{
	check(CSSM_SignDataInit(handle()));
	mStaged = true;
}

void Sign::sign(const CssmData *data, uint32 count)
{
	staged();
	check(CSSM_SignDataUpdate(handle(), data, count));
}

void Sign::operator () (CssmData &signature)
{
	staged();
	check(CSSM_SignDataFinal(handle(), &signature));
	mStaged = false;
}


//
// Verifying
//
void Verify::verify(const CssmData *data, uint32 count, const CssmData &signature)
{
	unstaged();
	check(CSSM_VerifyData(handle(), data, count, mSignOnly, &signature));
}

void Verify::init()
{
	check(CSSM_VerifyDataInit(handle()));
	mStaged = true;
}

void Verify::verify(const CssmData *data, uint32 count)
{
	staged();
	check(CSSM_VerifyDataUpdate(handle(), data, count));
}

void Verify::operator () (const CssmData &signature)
{
	staged();
	check(CSSM_VerifyDataFinal(handle(), &signature));
	mStaged = false;
}
