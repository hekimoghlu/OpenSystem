/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 2, 2022.
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
// Password.h - Password acquiring wrapper
//
#ifndef _SECURITY_PASSWORD_H_
#define _SECURITY_PASSWORD_H_

#include <security_keychain/Item.h>
// included by item #include <security_keychain/Keychains.h>
#include <security_keychain/Access.h>


namespace Security {
namespace KeychainCore {

class PasswordImpl : public SecCFObject {
public:
    SECCFFUNCTIONS(PasswordImpl, SecPasswordRef, errSecInvalidPasswordRef, gTypes().PasswordImpl)

public:
    // make default forms
    PasswordImpl(SecItemClass itemClass, SecKeychainAttributeList *searchAttrList, SecKeychainAttributeList *itemAttrList);
	PasswordImpl(PasswordImpl& existing);

    virtual ~PasswordImpl() _NOEXCEPT;

    bool getData(UInt32 *length, const void **data);
    void setData(UInt32 length,const void *data);
    void save();
    bool useKeychain() const { return mUseKeychain; }
    bool rememberInKeychain() const { return mRememberInKeychain; }
    void setRememberInKeychain(bool remember) { mRememberInKeychain = remember; }
    void setAccess(Access *access);

private:
    // keychain item cached?
    Keychain mKeychain;
    Item mItem;
    bool mUseKeychain;
    bool mFoundInKeychain;
    bool mRememberInKeychain;
	Mutex mMutex;
};

class Password : public SecPointer<PasswordImpl>
{
public:
    Password(SecItemClass itemClass, SecKeychainAttributeList *searchAttrList, SecKeychainAttributeList *itemAttrList);
    Password(PasswordImpl *impl);
	Password(PasswordImpl &impl);
};



            
} // end namespace KeychainCore
} // end namespace Security

#endif // !_SECURITY_PASSWORD_H_
