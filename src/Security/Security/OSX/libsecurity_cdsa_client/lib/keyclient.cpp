/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 7, 2023.
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
// keyclient
//
#include <security_cdsa_client/keyclient.h>
#include <security_cdsa_utilities/cssmdata.h>


using namespace CssmClient;


KeyImpl::KeyImpl(const CSP &csp) : ObjectImpl(csp), CssmKey() 
{
	mActive=false;
}

KeyImpl::KeyImpl(const CSP &csp, const CSSM_KEY &key, bool copy) : ObjectImpl(csp), CssmKey(key)
{
	if (copy)
		keyData() = CssmAutoData(csp.allocator(), keyData()).release();
	mActive=true;
}

KeyImpl::KeyImpl(const CSP &csp, const CSSM_DATA &keyData) : ObjectImpl(csp),
CssmKey((uint32)keyData.Length, csp->allocator().alloc<uint8>((UInt32)keyData.Length))
{
	memcpy(KeyData.Data, keyData.Data, keyData.Length);
	mActive=true;
}

KeyImpl::~KeyImpl()
try
{
    deactivate();
}
catch (...)
{
    return;	// Prevent re-throw of exception [function-try-block]
}

void
KeyImpl::deleteKey(const CSSM_ACCESS_CREDENTIALS *cred)
{
    StLock<Mutex> _(mActivateMutex);
	if (mActive)
	{
		mActive=false;
		check(CSSM_FreeKey(csp()->handle(), cred, this, CSSM_TRUE));
	}
}

CssmKeySize
KeyImpl::sizeInBits() const
{
    CssmKeySize size;
    check(CSSM_QueryKeySizeInBits(csp()->handle(), CSSM_INVALID_HANDLE, this, &size));
    return size;
}

void
KeyImpl::getAcl(AutoAclEntryInfoList &aclInfos, const char *selectionTag) const
{
	aclInfos.allocator(allocator());
	check(CSSM_GetKeyAcl(csp()->handle(), this, reinterpret_cast<const CSSM_STRING *>(selectionTag), aclInfos, aclInfos));
}

void
KeyImpl::changeAcl(const CSSM_ACL_EDIT &aclEdit,
	const CSSM_ACCESS_CREDENTIALS *accessCred)
{
	check(CSSM_ChangeKeyAcl(csp()->handle(),
		AccessCredentials::needed(accessCred), &aclEdit, this));
}

void
KeyImpl::getOwner(AutoAclOwnerPrototype &owner) const
{
	owner.allocator(allocator());
	check(CSSM_GetKeyOwner(csp()->handle(), this, owner));
}

void
KeyImpl::changeOwner(const CSSM_ACL_OWNER_PROTOTYPE &newOwner,
	const CSSM_ACCESS_CREDENTIALS *accessCred)
{
	check(CSSM_ChangeKeyOwner(csp()->handle(),
		AccessCredentials::needed(accessCred), this, &newOwner));
}

void KeyImpl::activate()
{
    StLock<Mutex> _(mActivateMutex);
	mActive=true;
}

void KeyImpl::deactivate()
{
    StLock<Mutex> _(mActivateMutex);
	if (mActive)
	{
		mActive=false;
		check(CSSM_FreeKey(csp()->handle(), NULL, this, CSSM_FALSE));
	}
}
