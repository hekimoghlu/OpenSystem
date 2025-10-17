/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 18, 2024.
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
 *	SecKeychainItemExtendedAttributes.h
 *	Created 9/6/06 by dmitch
 */

#ifndef _SEC_KEYCHAIN_ITEM_EXTENDED_ATTRIBUTES_H_
#define _SEC_KEYCHAIN_ITEM_EXTENDED_ATTRIBUTES_H_

#include <Security/SecBase.h>
#include <Security/cssmapple.h>
#include <CoreFoundation/CFArray.h>
#include <CoreFoundation/CFData.h>

#if defined(__cplusplus)
extern "C" {
#endif

/*
 * Extended attributes extend the fixed set of keychain item attribute in a generally
 * extensible way. A given SecKeychainItemRef can have assigned to it any number
 * of extended attributes, each consisting of an attribute name (as a CFStringRef)
 * and an attribute value (as a CFDataRef).
 *
 * Each extended attribute is a distinct record residing in the same keychain as
 * the item to which it refers. In a given keychain, the set of the following properties
 * of an extended attribute record must be unique:
 *
 *   -- the type of item to which the extended attribute is bound (kSecPublicKeyItemClass,
 *      kSecPrivateKeyItemClass, etc.)
 *   -- an identifier which uniquely identifies the item to which the extended attribute
 *      is bound. Currently this is the PrimaryKey blob.
 *   -- the extended attribute's Attribute Name, specified in this interface as a
 *      CFString.
 *
 * Thus, e.g., a given item can have at most one extended attribute with
 * Attribute Name of CFSTR("SomeAttributeName").
 */

/*
 * SecKeychainItemSetExtendedAttribute() - set an extended attribute by name and value.
 *
 * If the extended attribute specified by 'attrName' does not exist, one will be
 * created with the value specified in 'attrValue'.
 *
 * If the extended attribute specified by 'attrName already exists, its value will be
 * replaced by the value specified in 'attrValue'.
 *
 * If the incoming 'attrValue' is NULL, the extended attribute specified by 'attrName'
 * will be deleted if it exists. If the incoming 'attrValue' is NULL  and no such
 * attribute exists, the function will return errSecNoSuchAttr.
 */
OSStatus SecKeychainItemSetExtendedAttribute(
	SecKeychainItemRef			itemRef,
	CFStringRef					attrName,		/* identifies the attribute */
	CFDataRef					attrValue)		/* value to set; NULL means delete the
												 *    attribute */
API_DEPRECATED("SecKeychain is deprecated", macos(10.2, 10.10))
API_UNAVAILABLE(ios, watchos, tvos, bridgeos, macCatalyst);

/*
 * SecKeychainItemCopyExtendedAttribute() -  Obtain the value of an an extended attribute.
 *
 * If the extended attribute specified by 'attrName' exists, its value will be returned
 * via the *attrValue argument. The caller must CFRelease() this returned value.
 *
 * If the extended attribute specified by 'attrName' does not exist, the function
 * will return errSecNoSuchAttr.
 */
OSStatus SecKeychainItemCopyExtendedAttribute(
	SecKeychainItemRef			itemRef,
	CFStringRef					attrName,
	CFDataRef					*attrValue)
API_DEPRECATED("SecKeychain is deprecated", macos(10.2, 10.10))
API_UNAVAILABLE(ios, watchos, tvos, bridgeos, macCatalyst);

/*
 * SecKeychainItemCopyAllExtendedAttributes() - obtain all of an item's extended attributes.
 *
 * This is used to determine all of the extended attributes associated with a given
 * SecKeychainItemRef. The Atrribute Names of all of the extended attributes are
 * returned in the *attrNames argument; on successful return this contains a
 * CFArray whose elements are CFStringRefs, each of which is an Attribute Name.
 * The caller must CFRelease() this array.
 *
 * Optionally, the Attribute Values of all of the extended attributes is returned
 * in the *attrValues argument; on successful return this contains a CFArray whose
 * elements are CFDataRefs, each of which is an Attribute Value. The positions of
 * the elements in this array correspond with the elements in *attrNames; i.e.,
 * the n'th element in *attrName is the Attribute Name corresponding to the
 * Attribute Value found in the n'th element of *attrValues.
 *
 * Pass in NULL for attrValues if you don't need the Attribute Values. Caller
 * must CFRelease the array returned via this argument.
 *
 * If the item has no extended attributes, this function returns errSecNoSuchAttr.
 */
OSStatus SecKeychainItemCopyAllExtendedAttributes(
	SecKeychainItemRef			itemRef,
	CFArrayRef					*attrNames,			/* RETURNED, each element is a CFStringRef */
	CFArrayRef					*attrValues)		/* optional, RETURNED, each element is a
													 *   CFDataRef */
API_DEPRECATED("SecKeychain is deprecated", macos(10.2, 10.10))
API_UNAVAILABLE(ios, watchos, tvos, bridgeos, macCatalyst);

#if defined(__cplusplus)
}
#endif

#endif	/* _SEC_KEYCHAIN_ITEM_EXTENDED_ATTRIBUTES_H_ */

