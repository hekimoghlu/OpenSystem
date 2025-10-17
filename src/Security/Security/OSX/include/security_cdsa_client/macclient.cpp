/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 12, 2024.
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
// macclient - client interface to CSSM sign/verify mac contexts
//
#include <security_cdsa_client/macclient.h>

using namespace CssmClient;


//
// Common features of signing and verify mac contexts
//
void MacContext::activate()
{
    {
        StLock<Mutex> _(mActivateMutex);
        if (!mActive) 
        {
            check(CSSM_CSP_CreateMacContext(attachment()->handle(), mAlgorithm,
                  mKey, &mHandle));
            mActive = true;
        }
    }

    if (cred())
        cred(cred());		// install explicitly
}


//
// Signing
//
void GenerateMac::sign(const CssmData *data, uint32 count, CssmData &mac)
{
	unstaged();
	check(CSSM_GenerateMac(handle(), data, count, &mac));
}

void GenerateMac::init()
{
	check(CSSM_GenerateMacInit(handle()));
	mStaged = true;
}

void GenerateMac::sign(const CssmData *data, uint32 count)
{
	staged();
	check(CSSM_GenerateMacUpdate(handle(), data, count));
}

void GenerateMac::operator () (CssmData &mac)
{
	staged();
	check(CSSM_GenerateMacFinal(handle(), &mac));
	mStaged = false;
}


//
// Verifying
//
void VerifyMac::verify(const CssmData *data, uint32 count, const CssmData &mac)
{
	unstaged();
	check(CSSM_VerifyMac(handle(), data, count, &mac));
}

void VerifyMac::init()
{
	check(CSSM_VerifyMacInit(handle()));
	mStaged = true;
}

void VerifyMac::verify(const CssmData *data, uint32 count)
{
	staged();
	check(CSSM_VerifyMacUpdate(handle(), data, count));
}

void VerifyMac::operator () (const CssmData &mac)
{
	staged();
	check(CSSM_VerifyMacFinal(handle(), &mac));
	mStaged = false;
}
