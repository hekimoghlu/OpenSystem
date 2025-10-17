/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 11, 2025.
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
// clclient - client interface to CSSM CLs and their operations
//
#ifndef _H_CDSA_CLIENT_CLCLIENT
#define _H_CDSA_CLIENT_CLCLIENT  1

#include <security_cdsa_client/cssmclient.h>
#include <security_cdsa_utilities/cssmcert.h>


namespace Security {
namespace CssmClient {


//
// A CL attachment
//
class CLImpl : public AttachmentImpl
{
public:
	CLImpl(const Guid &guid);
	CLImpl(const Module &module);
	virtual ~CLImpl();
    
};

class CL : public Attachment
{
public:
	typedef CLImpl Impl;

	explicit CL(Impl *impl) : Attachment(impl) {}
	CL(const Guid &guid) : Attachment(new Impl(guid)) {}
	CL(const Module &module) : Attachment(new Impl(module)) {}

	Impl *operator ->() const { return &impl<Impl>(); }
	Impl &operator *() const { return impl<Impl>(); }
};


//
// A self-building CertGroup.
// This is a CertGroup, but it's NOT A PODWRAPPER (it's larger).
//
class BuildCertGroup : public CertGroup {
public:
    BuildCertGroup(CSSM_CERT_TYPE ctype, CSSM_CERT_ENCODING encoding,
        CSSM_CERTGROUP_TYPE type, Allocator &alloc = Allocator::standard());
    
    CssmVector<CSSM_DATA, CssmData> certificates;
};


} // end namespace CssmClient
} // end namespace Security

#endif // _H_CDSA_CLIENT_CLCLIENT
