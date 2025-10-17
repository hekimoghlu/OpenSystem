/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 16, 2024.
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
// TrustStore.h - Abstract interface to permanent user trust assignments
//
#include <security_keychain/TrustItem.h>
#include <security_cdsa_utilities/Schema.h>
#include <security_keychain/SecCFTypes.h>

#include <security_asn1/secasn1.h>
#include <security_asn1/SecNssCoder.h>
#include <Security/oidscert.h>


namespace Security {
namespace KeychainCore {


//
// Construct a UserTrustItem from attributes and initial content
//
UserTrustItem::UserTrustItem(Certificate *cert, Policy *policy, const TrustData &trustData) :
	ItemImpl((SecItemClass)CSSM_DL_DB_RECORD_USER_TRUST,
		reinterpret_cast<SecKeychainAttributeList *>(NULL),
		UInt32(sizeof(trustData)),
		reinterpret_cast<const void *>(&trustData)),
	mCertificate(cert), mPolicy(policy)
{
	secinfo("usertrust", "%p create(%p,%p) = %d",
		this, cert, policy, SecTrustUserSetting(trustData.trust));
}


//
// Destroy it
//
UserTrustItem::~UserTrustItem() 
{
	secinfo("usertrust", "%p destroyed", this);
}


//
// Retrieve the trust value from a UserTrustItem
//
UserTrustItem::TrustData UserTrustItem::trust()
{
	StLock<Mutex>_(mMutex);
	CssmDataContainer data;
	getData(data);
	if (data.length() != sizeof(TrustData))
		MacOSError::throwMe(errSecInvalidTrustSetting);
	return *data.interpretedAs<TrustData>();
}


//
// Add item to keychain
//
PrimaryKey UserTrustItem::add(Keychain &keychain)
{
	StLock<Mutex>_(mMutex);
	// If we already have a Keychain we can't be added.
	if (mKeychain)
		MacOSError::throwMe(errSecDuplicateItem);

	populateAttributes();

	CSSM_DB_RECORDTYPE recordType = mDbAttributes->recordType();

	Db db(keychain->database());
	// add the item to the (regular) db
	try
	{
		mUniqueId = db->insert(recordType, mDbAttributes.get(), mData.get());
		secinfo("usertrust", "%p inserted", this);
	}
	catch (const CssmError &e)
	{
		if (e.osStatus() != CSSMERR_DL_INVALID_RECORDTYPE)
			throw;

		// Create the trust relation and try again.
		secinfo("usertrust", "adding schema relation for user trusts");
		db->createRelation(CSSM_DL_DB_RECORD_USER_TRUST, "CSSM_DL_DB_RECORD_USER_TRUST",
			Schema::UserTrustSchemaAttributeCount,
			Schema::UserTrustSchemaAttributeList,
			Schema::UserTrustSchemaIndexCount,
			Schema::UserTrustSchemaIndexList);
		keychain->keychainSchema()->didCreateRelation(
			CSSM_DL_DB_RECORD_USER_TRUST,
			"CSSM_DL_DB_RECORD_USER_TRUST",
			Schema::UserTrustSchemaAttributeCount,
			Schema::UserTrustSchemaAttributeList,
			Schema::UserTrustSchemaIndexCount,
			Schema::UserTrustSchemaIndexList);

		mUniqueId = db->insert(recordType, mDbAttributes.get(), mData.get());
		secinfo("usertrust", "%p inserted now", this);
	}

	mPrimaryKey = keychain->makePrimaryKey(recordType, mUniqueId);
    mKeychain = keychain;
	return mPrimaryKey;
}


void UserTrustItem::populateAttributes()
{
	StLock<Mutex>_(mMutex);
	CssmAutoData encodedIndex(Allocator::standard());
	makeCertIndex(mCertificate, encodedIndex);
	const CssmOid &policyOid = mPolicy->oid();

	mDbAttributes->add(Schema::attributeInfo(kSecTrustCertAttr), encodedIndex.get());
	mDbAttributes->add(Schema::attributeInfo(kSecTrustPolicyAttr), policyOid);
}


//
// An ad-hoc hold-and-destroy accessor for a single-valued certificate field
//
class CertField {
public:
	CertField(Certificate *cert, const CSSM_OID &inField)
		: certificate(cert), field(inField)
	{ mData = certificate->copyFirstFieldValue(field); }
		
	~CertField() { certificate->releaseFieldValue(field, mData); }
	
	Certificate * const certificate;
	const CSSM_OID &field;
	
	operator bool () const { return mData && mData->Data; }
	CssmData &data() const { return CssmData::overlay(*mData); }

private:
	CSSM_DATA_PTR mData;
};


//
// Construct a trust item index.
// This is an ASN.1 sequence of issuer and serial number.
//
struct IssuerAndSN {
    CSSM_DATA issuer;
    CSSM_DATA serial;
};

static const SecAsn1Template issuerAndSNTemplate[] = {
	{ SEC_ASN1_SEQUENCE, 0, NULL, sizeof(IssuerAndSN) },
	{ SEC_ASN1_OCTET_STRING, offsetof(IssuerAndSN, issuer) },
	{ SEC_ASN1_OCTET_STRING, offsetof(IssuerAndSN, serial) },
	{ 0 }
};

void UserTrustItem::makeCertIndex(Certificate *cert, CssmOwnedData &encodedIndex)
{
	CertField issuer(cert, CSSMOID_X509V1IssuerName);
	CertField serial(cert, CSSMOID_X509V1SerialNumber);
	IssuerAndSN index;
	index.issuer = issuer.data();
	index.serial = serial.data();
	if (SecNssEncodeItemOdata(&index, issuerAndSNTemplate, encodedIndex))
		CssmError::throwMe(CSSMERR_CSP_MEMORY_ERROR);
}


} // end namespace KeychainCore
} // end namespace Security
