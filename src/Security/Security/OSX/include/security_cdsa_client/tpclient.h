/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 23, 2023.
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
#ifndef _H_CDSA_CLIENT_TPCLIENT
#define _H_CDSA_CLIENT_TPCLIENT  1

#include <security_cdsa_client/cssmclient.h>
#include <security_cdsa_client/clclient.h>
#include <security_cdsa_client/cspclient.h>
#include <security_cdsa_utilities/cssmtrust.h>
#include <security_cdsa_utilities/cssmalloc.h>
#include <security_cdsa_utilities/cssmdata.h>


namespace Security {
namespace CssmClient {


//
// A TP attachment
//
class TPImpl : public AttachmentImpl
{
public:
	TPImpl(const Guid &guid);
	TPImpl(const Module &module);
	virtual ~TPImpl();
    
public:
    // the CL and CSP used with many TP operations is usually
    // pretty stable. The system may even figure them out
    // automatically in the future.
    void use(CL &cl);
    void use(CSP &csp);
    CL &usedCL();
    CSP &usedCSP();

public:
    void certGroupVerify(const CertGroup &certGroup, const TPVerifyContext &context,
        TPVerifyResult *result);

private:
    void setupCL();				// setup mUseCL
    void setupCSP();			// setup mUseCSP

private:
    CL *mUseCL;				// use this CL for TP operation
    CSP *mUseCSP;			// use this CSP for TP operation
    bool mOwnCL, mOwnCSP;	// whether we've made our own
};


class TP : public Attachment
{
public:
	typedef TPImpl Impl;

	explicit TP(Impl *impl) : Attachment(impl) {}
	TP(const Guid &guid) : Attachment(new Impl(guid)) {}
	TP(const Module &module) : Attachment(new Impl(module)) {}

	Impl *operator ->() const { return &impl<Impl>(); }
	Impl &operator *() const { return impl<Impl>(); }
};


//
// A self-building TPVerifyContext.
// This is a TPVerifyContext, but it's NOT A PODWRAPPER (it's larger).
//
// NOTE: This is not a client-side object.
//
class TPBuildVerifyContext : public TPVerifyContext {
public:
    TPBuildVerifyContext(CSSM_TP_ACTION action = CSSM_TP_ACTION_DEFAULT,
        Allocator &alloc = Allocator::standard());
    
    Allocator &allocator;
    
private:
    TPCallerAuth mCallerAuth;
    // PolicyInfo mPolicyInfo; // -- unused
	CssmDlDbList mDlDbList;
};


} // end namespace CssmClient
} // end namespace Security

#endif // _H_CDSA_CLIENT_CLCLIENT
