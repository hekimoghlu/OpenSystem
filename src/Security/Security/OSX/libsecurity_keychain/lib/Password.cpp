/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 8, 2024.
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
// Password.cpp
//
#include "Password.h"
#include <Security/SecBase.h>
#include "SecBridge.h"

#include "KCCursor.h"

using namespace KeychainCore;
using namespace CssmClient;

PasswordImpl::PasswordImpl(SecItemClass itemClass, SecKeychainAttributeList *searchAttrList, SecKeychainAttributeList *itemAttrList) :
    mItem(itemClass, itemAttrList, 0, NULL), mUseKeychain(false), mFoundInKeychain(false), mRememberInKeychain(false), mMutex(Mutex::recursive)
{
    if (searchAttrList && itemAttrList)
    {
        mUseKeychain = true;
        mKeychain = Keychain::optional(NULL);
		mRememberInKeychain = true;

        // initialize mFoundInKeychain to true if mItem is found
        
        StorageManager::KeychainList keychains;
        globals().storageManager.optionalSearchList(NULL, keychains);
        KCCursor cursor(keychains, itemClass, searchAttrList);

        if (cursor->next(mItem))
            mFoundInKeychain = true;
    }
}

PasswordImpl::PasswordImpl(PasswordImpl& existing)
{
	mKeychain = existing.mKeychain;
	mItem = existing.mItem;
    mUseKeychain = existing.mUseKeychain;
    mFoundInKeychain = existing.mFoundInKeychain;
    mRememberInKeychain = existing.mRememberInKeychain;
}



PasswordImpl::~PasswordImpl() _NOEXCEPT
{
}

void
PasswordImpl::setAccess(Access *access)
{
    // changing an existing ACL is more work than this SPI wants to do
    if (!mFoundInKeychain)
        mItem->setAccess(access);
}

void
PasswordImpl::setData(UInt32 length, const void *data)
{
    assert(mUseKeychain);
    
    // do different things based on mFoundInKeychain?
    mItem->setData(length,data);
}

bool
PasswordImpl::getData(UInt32 *length, const void **data)
{
    if (mItem->isPersistent())
    {
        // try to retrieve it
        CssmDataContainer outData;
        try
        {
            mItem->getData(outData);
            if (length && data)
            {
                *length=(uint32)outData.length();
                outData.Length=0;
                *data=outData.data();
                outData.Data=NULL;
            }
            return true;
        }
        catch (...)
        {
            // cancel unlock: CSP_USER_CANCELED
            // deny rogue app CSP_OPERATION_AUTH_DENIED
            return false;
        }
    }
    else
        return false;
}

void
PasswordImpl::save()
{
    assert(mUseKeychain);
    
    if (mFoundInKeychain)
    {
        mItem->update();
    }
    else
    {
        mKeychain->add(mItem);

        // reinitialize mItem now it's on mKeychain
        mFoundInKeychain = true; // should be set by member that resets mItem
    }
}

Password::Password(SecItemClass itemClass, SecKeychainAttributeList *searchAttrList, SecKeychainAttributeList *itemAttrList) : 
    SecPointer<PasswordImpl>(new PasswordImpl(itemClass, searchAttrList, itemAttrList))
{
}

Password::Password(PasswordImpl *impl) : SecPointer<PasswordImpl>(impl)
{
}

Password::Password(PasswordImpl &impl) : SecPointer<PasswordImpl>(new PasswordImpl(impl))
{
}
