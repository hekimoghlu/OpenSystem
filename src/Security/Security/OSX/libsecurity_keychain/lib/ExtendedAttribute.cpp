/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 5, 2023.
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
/*
 * ExtendedAttribute.cpp - Extended Keychain Item Attribute class.
 *
 */

#include "ExtendedAttribute.h"
#include "SecKeychainItemExtendedAttributes.h"
#include "SecKeychainItemPriv.h"
#include "cssmdatetime.h"
#include <security_cdsa_utilities/Schema.h>

using namespace KeychainCore;

/* 
 * Construct new ExtendedAttr from API.
 */
ExtendedAttribute::ExtendedAttribute(
	CSSM_DB_RECORDTYPE recordType, 
	const CssmData &itemID, 
	const CssmData attrName,
	const CssmData attrValue) :
		ItemImpl((SecItemClass) CSSM_DL_DB_RECORD_EXTENDED_ATTRIBUTE,
			reinterpret_cast<SecKeychainAttributeList *>(NULL), 
			0, NULL),
		mRecordType(recordType),
		mItemID(Allocator::standard(), itemID.Data, itemID.Length),
		mAttrName(Allocator::standard(), attrName.Data, attrName.Length),
		mAttrValue(Allocator::standard(), attrValue.Data, attrValue.Length)
{
	setupAttrs();
}

// db item contstructor
ExtendedAttribute::ExtendedAttribute(
	const Keychain &keychain, 
	const PrimaryKey &primaryKey, 
	const CssmClient::DbUniqueRecord &uniqueId) :
		ItemImpl(keychain, primaryKey, uniqueId),
		mRecordType(0),
		mItemID(Allocator::standard()),
		mAttrName(Allocator::standard()),
		mAttrValue(Allocator::standard())
{

}

// PrimaryKey item contstructor
ExtendedAttribute::ExtendedAttribute(
	const Keychain &keychain, 
	const PrimaryKey &primaryKey) :
		ItemImpl(keychain, primaryKey),
		mRecordType(0),
		mItemID(Allocator::standard()),
		mAttrName(Allocator::standard()),
		mAttrValue(Allocator::standard())
{

}

ExtendedAttribute* ExtendedAttribute::make(const Keychain &keychain, const PrimaryKey &primaryKey, const CssmClient::DbUniqueRecord &uniqueId)
{
	ExtendedAttribute* ea = new ExtendedAttribute(keychain, primaryKey, uniqueId);
	keychain->addItem(primaryKey, ea);
	return ea;
}



ExtendedAttribute* ExtendedAttribute::make(const Keychain &keychain, const PrimaryKey &primaryKey)
{
	ExtendedAttribute* ea = new ExtendedAttribute(keychain, primaryKey);
	keychain->addItem(primaryKey, ea);
	return ea;
}



// copy - required due to Item's weird constructor/vendor
ExtendedAttribute::ExtendedAttribute(
	ExtendedAttribute &extendedAttr) :
		ItemImpl(extendedAttr),
		mRecordType(extendedAttr.mRecordType),
		mItemID(Allocator::standard()),
		mAttrName(Allocator::standard()),
		mAttrValue(Allocator::standard())
{
	// CssmData cd = extendedAttr.mItemID;
	mItemID.copy(extendedAttr.mItemID);
	// cd = extendedAttr.mAttrName;
	mAttrName.copy(extendedAttr.mAttrName);
	// cd = extendedAttr.mAttrValue;
	mAttrValue.copy(extendedAttr.mAttrValue);
	setupAttrs();
}

ExtendedAttribute::~ExtendedAttribute() _NOEXCEPT
{

}

PrimaryKey
ExtendedAttribute::add(Keychain &keychain)
{
	StLock<Mutex>_(mMutex);
	// If we already have a Keychain we can't be added.
	if (mKeychain)
		MacOSError::throwMe(errSecDuplicateItem);

	SInt64 date;
	CSSMDateTimeUtils::GetCurrentMacLongDateTime(date);
	CssmDbAttributeInfo attrInfo(kSecModDateItemAttr, CSSM_DB_ATTRIBUTE_FORMAT_TIME_DATE);
	setAttribute(attrInfo, date);

	Db db(keychain->database());
	// add the item to the (regular) db
	try
	{
		mUniqueId = db->insert(CSSM_DL_DB_RECORD_EXTENDED_ATTRIBUTE, mDbAttributes.get(), mData.get());
	}
	catch (const CssmError &e)
	{
		if (e.osStatus() != CSSMERR_DL_INVALID_RECORDTYPE)
			throw;

		/* 
		 * First exposure of this keychain to the extended attribute record type.
		 * Create the relation and try again.
		 */
		db->createRelation(CSSM_DL_DB_RECORD_EXTENDED_ATTRIBUTE,
			"CSSM_DL_DB_RECORD_EXTENDED_ATTRIBUTE",
			Schema::ExtendedAttributeSchemaAttributeCount,
			Schema::ExtendedAttributeSchemaAttributeList,
			Schema::ExtendedAttributeSchemaIndexCount,
			Schema::ExtendedAttributeSchemaIndexList);
		keychain->keychainSchema()->didCreateRelation(
			CSSM_DL_DB_RECORD_EXTENDED_ATTRIBUTE,
			"CSSM_DL_DB_RECORD_EXTENDED_ATTRIBUTE",
			Schema::ExtendedAttributeSchemaAttributeCount,
			Schema::ExtendedAttributeSchemaAttributeList,
			Schema::ExtendedAttributeSchemaIndexCount,
			Schema::ExtendedAttributeSchemaIndexList);

		mUniqueId = db->insert(CSSM_DL_DB_RECORD_EXTENDED_ATTRIBUTE, mDbAttributes.get(), mData.get());
	}

	mPrimaryKey = keychain->makePrimaryKey(CSSM_DL_DB_RECORD_EXTENDED_ATTRIBUTE, mUniqueId);
    mKeychain = keychain;

	return mPrimaryKey;
}

/* set up DB attrs based on member vars */
void ExtendedAttribute::setupAttrs()
{
	StLock<Mutex>_(mMutex);
	CssmDbAttributeInfo attrInfo1(kExtendedAttrRecordTypeAttr, CSSM_DB_ATTRIBUTE_FORMAT_UINT32);
	setAttribute(attrInfo1, (uint32)mRecordType);
	CssmData cd = mItemID;
	CssmDbAttributeInfo attrInfo2(kExtendedAttrItemIDAttr, CSSM_DB_ATTRIBUTE_FORMAT_BLOB);
	setAttribute(attrInfo2, cd);
	cd = mAttrName;
	CssmDbAttributeInfo attrInfo3(kExtendedAttrAttributeNameAttr, CSSM_DB_ATTRIBUTE_FORMAT_BLOB);
	setAttribute(attrInfo3, cd);
	cd = mAttrValue;
	CssmDbAttributeInfo attrInfo4(kExtendedAttrAttributeValueAttr, CSSM_DB_ATTRIBUTE_FORMAT_BLOB);
	setAttribute(attrInfo4, cd);
}


