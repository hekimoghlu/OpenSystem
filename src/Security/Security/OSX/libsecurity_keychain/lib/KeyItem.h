/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 27, 2024.
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
// KeyItem.h
//
#ifndef _SECURITY_KEYITEM_H_
#define _SECURITY_KEYITEM_H_

#include <security_keychain/Item.h>
#include <Security/SecKeyPriv.h>

namespace Security
{

namespace KeychainCore
{

class KeyItem : public ItemImpl
{
	NOCOPY(KeyItem)
public:
	SECCFFUNCTIONS_BASE(KeyItem, SecKeyRef)

    // SecKeyRef is now provided by iOS implementation, so we have to hack standard accessors normally defined by
    // SECCFUNCTIONS macro to retarget SecKeyRef to foreign object instead of normal way through SecCFObject.
    static KeyItem *required(SecKeyRef ptr);
    static KeyItem *optional(SecKeyRef ptr);
    operator CFTypeRef() const _NOEXCEPT;
    static SecCFObject *fromSecKeyRef(CFTypeRef ref);
    void attachSecKeyRef() const;
    void initializeWithSecKeyRef(SecKeyRef ref);

private:
    // This weak backpointer to owning SecKeyRef instance (which is created by iOS SecKey code).
    mutable SecKeyRef mWeakSecKeyRef;

	// db item constructor
private:
    KeyItem(const Keychain &keychain, const PrimaryKey &primaryKey, const CssmClient::DbUniqueRecord &uniqueId);

	// PrimaryKey item constructor
    KeyItem(const Keychain &keychain, const PrimaryKey &primaryKey);

public:
	static KeyItem* make(const Keychain &keychain, const PrimaryKey &primaryKey, const CssmClient::DbUniqueRecord &uniqueId);
	static KeyItem* make(const Keychain &keychain, const PrimaryKey &primaryKey);
	
	KeyItem(KeyItem &keyItem);

	KeyItem(const CssmClient::Key &key);

    virtual ~KeyItem();

	virtual void update();
	virtual Item copyTo(const Keychain &keychain, Access *newAccess = NULL);
	virtual Item importTo(const Keychain &keychain, Access *newAccess = NULL, SecKeychainAttributeList *attrList = NULL);
	virtual void didModify();

	CssmClient::SSDbUniqueRecord ssDbUniqueRecord();
	CssmClient::Key &key();
	CssmClient::CSP csp();

    // Returns the header of the unverified key (without checking integrity). This will skip ACL checks, but don't trust the data very much.
    // Can't return a reference, because maybe the unverified key will get released upon return.
    CssmKey::Header unverifiedKeyHeader();

	const CSSM_X509_ALGORITHM_IDENTIFIER& algorithmIdentifier();
	unsigned int strengthInBits(const CSSM_X509_ALGORITHM_IDENTIFIER *algid);
    CssmClient::Key publicKey();

	const AccessCredentials *getCredentials(
		CSSM_ACL_AUTHORIZATION_TAG operation,
		SecCredentialType credentialType);

	bool operator == (KeyItem &other);

	static void createPair(
		Keychain keychain,
        CSSM_ALGORITHMS algorithm,
        uint32 keySizeInBits,
        CSSM_CC_HANDLE contextHandle,
        CSSM_KEYUSE publicKeyUsage,
        uint32 publicKeyAttr,
        CSSM_KEYUSE privateKeyUsage,
        uint32 privateKeyAttr,
        SecPointer<Access> initialAccess,
        SecPointer<KeyItem> &outPublicKey, 
        SecPointer<KeyItem> &outPrivateKey);

	static void importPair(
		Keychain keychain,
		const CSSM_KEY &publicCssmKey,
		const CSSM_KEY &privateCssmKey,
        SecPointer<Access> initialAccess,
        SecPointer<KeyItem> &outPublicKey, 
        SecPointer<KeyItem> &outPrivateKey);

	static SecPointer<KeyItem> generate(
		Keychain keychain,
		CSSM_ALGORITHMS algorithm,
		uint32 keySizeInBits,
		CSSM_CC_HANDLE contextHandle,
		CSSM_KEYUSE keyUsage,
		uint32 keyAttr,
		SecPointer<Access> initialAccess);

	static SecPointer<KeyItem> generateWithAttributes(
		const SecKeychainAttributeList *attrList,
		Keychain keychain,
		CSSM_ALGORITHMS algorithm,
		uint32 keySizeInBits,
		CSSM_CC_HANDLE contextHandle,
		CSSM_KEYUSE keyUsage,
		uint32 keyAttr,
		SecPointer<Access> initialAccess);

	virtual const CssmData &itemID();
	
	virtual CFHashCode hash();

    virtual void setIntegrity(bool force = false);
    virtual bool checkIntegrity();

    // Call this function to remove the integrity and partition_id ACLs from
    // this item. You're not supposed to be able to do this, so force the issue
    // by providing credentials to this keychain.
    virtual void removeIntegrity(const AccessCredentials *cred);

    static void modifyUniqueId(Keychain keychain, SSDb ssDb, DbUniqueRecord& uniqueId, DbAttributes& newDbAttributes, CSSM_DB_RECORDTYPE recordType);

protected:
	virtual PrimaryKey add(Keychain &keychain);
private:
    CssmClient::Key unverifiedKey();

	CssmClient::Key mKey;
	const CSSM_X509_ALGORITHM_IDENTIFIER *algid;
	CssmAutoData mPubKeyHash;
    CssmClient::Key mPublicKey;
};

} // end namespace KeychainCore

} // end namespace Security

class CDSASecKey : public __SecKey {
public:
    static Security::KeychainCore::KeyItem *keyItem(SecKeyRef key) {
        CDSASecKey *cdsaKey = static_cast<CDSASecKey *>(key);
        return static_cast<Security::KeychainCore::KeyItem *>(cdsaKey->key);
    }
    SecCredentialType credentialType;
    Mutex *cdsaKeyMutex;
};

#endif // !_SECURITY_KEYITEM_H_
