/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 31, 2023.
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
// UnlockReferralItem - Abstract interface to permanent user trust assignments
//
#include <security_keychain/UnlockReferralItem.h>
#include <security_cdsa_utilities/Schema.h>
#include <security_keychain/SecCFTypes.h>


namespace Security {
namespace KeychainCore {


//
// Construct a UnlockReferralItem from attributes and initial content
//
UnlockReferralItem::UnlockReferralItem() :
	ItemImpl((SecItemClass) CSSM_DL_DB_RECORD_UNLOCK_REFERRAL,
		reinterpret_cast<SecKeychainAttributeList *>(NULL),
		UInt32(0/*size*/),
		NULL/*data*/)
{
	secinfo("referral", "create %p", this);
}


//
// Destroy it
//
UnlockReferralItem::~UnlockReferralItem() 
{
	secinfo("referral", "destroy %p", this);
}


//
// Add item to keychain
//
PrimaryKey UnlockReferralItem::add(Keychain &keychain)
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

		// Create the referral relation and try again.
		secinfo("usertrust", "adding schema relation for user trusts");
#if 0
		db->createRelation(CSSM_DL_DB_RECORD_UNLOCK_REFERRAL,
			"CSSM_DL_DB_RECORD_UNLOCK_REFERRAL",
			Schema::UnlockReferralSchemaAttributeCount,
			Schema::UnlockReferralSchemaAttributeList,
			Schema::UnlockReferralSchemaIndexCount,
			Schema::UnlockReferralSchemaIndexList);
		keychain->keychainSchema()->didCreateRelation(
			CSSM_DL_DB_RECORD_UNLOCK_REFERRAL,
			"CSSM_DL_DB_RECORD_UNLOCK_REFERRAL",
			Schema::UnlockReferralSchemaAttributeCount,
			Schema::UnlockReferralSchemaAttributeList,
			Schema::UnlockReferralSchemaIndexCount,
			Schema::UnlockReferralSchemaIndexList);
#endif
		//keychain->resetSchema();

		mUniqueId = db->insert(recordType, mDbAttributes.get(), mData.get());
		secinfo("usertrust", "%p inserted now", this);
	}

	mPrimaryKey = keychain->makePrimaryKey(recordType, mUniqueId);
    mKeychain = keychain;
	return mPrimaryKey;
}


void UnlockReferralItem::populateAttributes()
{
#if 0
	CssmAutoData encodedIndex(Allocator::standard());
	makeCertIndex(mCertificate, encodedIndex);
	const CssmOid &policyOid = mPolicy->oid();

	mDbAttributes->add(Schema::attributeInfo(kSecTrustCertAttr), encodedIndex.get());
	mDbAttributes->add(Schema::attributeInfo(kSecTrustPolicyAttr), policyOid);
#endif
}


} // end namespace KeychainCore
} // end namespace Security
