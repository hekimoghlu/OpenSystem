/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 22, 2023.
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
// tpclient - client interface to CSSM TPs and their operations
//
#include <security_cdsa_client/tpclient.h>

namespace Security {
namespace CssmClient {


//
// Manage TP attachments
//
TPImpl::TPImpl(const Guid &guid)
    : AttachmentImpl(guid, CSSM_SERVICE_TP), mUseCL(NULL), mUseCSP(NULL),
    mOwnCL(false), mOwnCSP(false)
{
}

TPImpl::TPImpl(const Module &module)
    : AttachmentImpl(module, CSSM_SERVICE_TP), mUseCL(NULL), mUseCSP(NULL),
    mOwnCL(false), mOwnCSP(false)
{
}

TPImpl::~TPImpl()
{
    if (mOwnCL)
        delete mUseCL;
    if (mOwnCSP)
        delete mUseCSP;
}


//
// Verify a CertGroup
//
void TPImpl::certGroupVerify(const CertGroup &certGroup,
    const TPVerifyContext &context,
    TPVerifyResult *result)
{
    setupCL();
    setupCSP();
    check(CSSM_TP_CertGroupVerify(handle(), (*mUseCL)->handle(), (*mUseCSP)->handle(),
        &certGroup, &context, result));
}


//
// Initialize auxiliary modules for operation
//
void TPImpl::setupCL()
{
    if (mUseCL == NULL) {
        secinfo("tpclient", "TP is auto-attaching supporting CL");
        mUseCL = new CL(gGuidAppleX509CL);
        mOwnCL = true;
    }
}

void TPImpl::setupCSP()
{
    if (mUseCSP == NULL) {
        secinfo("tpclient", "TP is auto-attaching supporting CSP");
        mUseCSP = new CSP(gGuidAppleCSP);
        mOwnCSP = true;
    }
}

void TPImpl::use(CL &cl)
{
    if (mOwnCL)
        delete mUseCL;
    mUseCL = &cl;
    mOwnCL = false;
}

void TPImpl::use(CSP &csp)
{
    if (mOwnCSP)
        delete mUseCSP;
    mUseCSP = &csp;
    mOwnCSP = false;
}

CL &TPImpl::usedCL()
{
    setupCL();
    return *mUseCL;
}

CSP &TPImpl::usedCSP()
{
    setupCSP();
    return *mUseCSP;
}


//
// A TPBuildVerifyContext
//
TPBuildVerifyContext::TPBuildVerifyContext(CSSM_TP_ACTION action, Allocator &alloc)
    : allocator(alloc)
{
    // clear out the PODs
    clearPod();
    mCallerAuth.clearPod();
	mDlDbList.clearPod();
    
    // set initial elements
    Action = action;
    callerAuthPtr(&mCallerAuth);
	mCallerAuth.dlDbList() = &mDlDbList;
}


}	// end namespace CssmClient
}	// end namespace Security

