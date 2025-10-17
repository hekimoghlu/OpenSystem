/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 26, 2022.
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
// Certificate.h - Certificate objects
//
#ifndef _SECURITY_CERTIFICATE_H_
#define _SECURITY_CERTIFICATE_H_

#include <security_keychain/Item.h>

#include <security_keychain/StorageManager.h>
// @@@ This should not be here.
#include <Security/SecBase.h>
#include <security_cdsa_client/clclient.h>

namespace Security
{

namespace KeychainCore
{

class KeyItem;

class Certificate : public ItemImpl
{
	NOCOPY(Certificate)
public:
	SECCFFUNCTIONS(Certificate, SecCertificateRef, errSecInvalidItemRef, gTypes().Certificate)

	static CL clForType(CSSM_CERT_TYPE type);

	// new item constructor
    Certificate(const CSSM_DATA &data, CSSM_CERT_TYPE type, CSSM_CERT_ENCODING encoding);

private:
	// db item constructor
    Certificate(const Keychain &keychain, const PrimaryKey &primaryKey, const CssmClient::DbUniqueRecord &uniqueId);

	// PrimaryKey item constructor
    Certificate(const Keychain &keychain, const PrimaryKey &primaryKey);

public:
	static Certificate* make(const Keychain &keychain, const PrimaryKey &primaryKey, const CssmClient::DbUniqueRecord &uniqueId);
	static Certificate* make(const Keychain &keychain, const PrimaryKey &primaryKey);

	Certificate(Certificate &certificate);
    virtual ~Certificate();

	virtual void update();
	virtual Item copyTo(const Keychain &keychain, Access *newAccess = NULL);
	virtual void didModify(); // Forget any attributes and data we just wrote to the db

    const CssmData &data();
    CSSM_CERT_TYPE type();
	CSSM_CERT_ENCODING encoding();
    CFDataRef sha1Hash();
    CFDataRef sha256Hash();
	CFStringRef commonName();
	CFStringRef distinguishedName(const CSSM_OID *sourceOid, const CSSM_OID *componentOid);
	CFStringRef copyFirstEmailAddress();
	CFArrayRef copyEmailAddresses();
	CFArrayRef copyDNSNames();
    CSSM_X509_NAME_PTR subjectName();
    CSSM_X509_NAME_PTR issuerName();
	CSSM_X509_ALGORITHM_IDENTIFIER_PTR algorithmID();
   	CSSM_CL_HANDLE clHandle();
	void inferLabel(bool addLabel, CFStringRef *rtnString = NULL);
	SecPointer<KeyItem> publicKey();
	const CssmData &publicKeyHash();
	const CssmData &subjectKeyIdentifier();

	static KCCursor cursorForIssuerAndSN(const StorageManager::KeychainList &keychains, const CssmData &issuer, const CssmData &serialNumber);
	static KCCursor cursorForSubjectKeyID(const StorageManager::KeychainList &keychains, const CssmData &subjectKeyID);
	static KCCursor cursorForEmail(const StorageManager::KeychainList &keychains, const char *emailAddress);
	static KCCursor cursorForIssuerAndSN_CF(const StorageManager::KeychainList &keychains, CFDataRef issuer, CFDataRef serialNumber);

	SecPointer<Certificate> findInKeychain(const StorageManager::KeychainList &keychains);
	static SecPointer<Certificate> findByIssuerAndSN(const StorageManager::KeychainList &keychains, const CssmData &issuer, const CssmData &serialNumber);
	static SecPointer<Certificate> findBySubjectKeyID(const StorageManager::KeychainList &keychains, const CssmData &subjectKeyID);
	static SecPointer<Certificate> findByEmail(const StorageManager::KeychainList &keychains, const char *emailAddress);

	static void normalizeEmailAddress(CSSM_DATA &emailAddress);
	static void getNames(CSSM_DATA_PTR *sanValues, CSSM_DATA_PTR snValue, CE_GeneralNameType generalNameType, std::vector<CssmData> &names);

	bool operator < (Certificate &other);
	bool operator == (Certificate &other);

	virtual CFHashCode hash();

public:
	CSSM_DATA_PTR copyFirstFieldValue(const CSSM_OID &field);
	void releaseFieldValue(const CSSM_OID &field, CSSM_DATA_PTR fieldValue);

	CSSM_DATA_PTR *copyFieldValues(const CSSM_OID &field);
	void releaseFieldValues(const CSSM_OID &field, CSSM_DATA_PTR *fieldValues);
	Boolean isSelfSigned();

protected:
	virtual void willRead();
	virtual PrimaryKey add(Keychain &keychain);
	CSSM_HANDLE certHandle();

	void addParsedAttribute(const CSSM_DB_ATTRIBUTE_INFO &info, const CSSM_OID &field);

	void addSubjectKeyIdentifier();
	void populateAttributes();
	bool verifyEncoding(CSSM_DATA_PTR data);

private:
	bool mHaveTypeAndEncoding;
	bool mPopulated;
    CSSM_CERT_TYPE mType;
	CSSM_CERT_ENCODING mEncoding;
    CssmClient::CL mCL;
	CSSM_HANDLE mCertHandle;
	CssmData mPublicKeyHash;
	uint8 mPublicKeyHashBytes[20];
	CssmData mSubjectKeyID;
	uint8 mSubjectKeyIDBytes[20];
	CSSM_DATA_PTR mV1SubjectPublicKeyCStructValue; // Hack to prevent algorithmID() from leaking.
    CSSM_DATA_PTR mV1SubjectNameCStructValue;
    CSSM_DATA_PTR mV1IssuerNameCStructValue;
    CFDataRef mSha1Hash;
    CFDataRef mSha256Hash;
	bool mEncodingVerified;
};

} // end namespace KeychainCore

} // end namespace Security

#endif // !_SECURITY_CERTIFICATE_H_
