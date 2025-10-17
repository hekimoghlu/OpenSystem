/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 18, 2024.
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
 * ExtendedAttribute.h - Extended Keychain Item Attribute class.
 *
 */

#ifndef _SECURITY_EXTENDED_ATTRIBUTE_H_
#define _SECURITY_EXTENDED_ATTRIBUTE_H_

#include <security_keychain/Item.h>
#include <security_cdsa_utilities/cssmdata.h>

/* this is not public */
typedef struct OpaqueSecExtendedAttributeRef *SecKeychainItemExtendedAttributeRef;

namespace Security
{

namespace KeychainCore
{

class ExtendedAttribute : public ItemImpl
{
	NOCOPY(ExtendedAttribute)
public:
	SECCFFUNCTIONS(ExtendedAttribute, SecKeychainItemExtendedAttributeRef, 
		errSecInvalidItemRef, gTypes().ExtendedAttribute)

	/* construct new ExtendedAttr from API */
	ExtendedAttribute(CSSM_DB_RECORDTYPE recordType, 
		const CssmData &itemID, 
		const CssmData attrName,
		const CssmData attrValue);

private:
	// db item contstructor
    ExtendedAttribute(const Keychain &keychain, 
		const PrimaryKey &primaryKey, 
		const CssmClient::DbUniqueRecord &uniqueId);

	// PrimaryKey item contstructor
    ExtendedAttribute(const Keychain &keychain, const PrimaryKey &primaryKey);

public:
	static ExtendedAttribute* make(const Keychain &keychain, const PrimaryKey &primaryKey, const CssmClient::DbUniqueRecord &uniqueId);
	static ExtendedAttribute* make(const Keychain &keychain, const PrimaryKey &primaryKey);
	
	ExtendedAttribute(ExtendedAttribute &extendedAttribute);

    virtual ~ExtendedAttribute() _NOEXCEPT;

	virtual PrimaryKey add(Keychain &keychain);
	bool operator == (const ExtendedAttribute &other) const;
private:
	/* set up DB attrs based on member vars */
	void setupAttrs();
	
	CSSM_DB_RECORDTYPE		mRecordType;
	CssmAutoData			mItemID;
	CssmAutoData			mAttrName;
	CssmAutoData			mAttrValue;
};

} // end namespace KeychainCore

} // end namespace Security

#endif /* _SECURITY_EXTENDED_ATTRIBUTES_H_ */
