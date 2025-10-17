/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 28, 2022.
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
/*!
	@header SecKeychainSearch
	The functions provided in SecKeychainSearch implement a query of one or more keychains to search for a particular SecKeychainItem.
*/

#ifndef _SECURITY_SECKEYCHAINSEARCH_H_
#define _SECURITY_SECKEYCHAINSEARCH_H_

#include <Security/SecKeychainItem.h>


#if defined(__cplusplus)
extern "C" {
#endif

CF_ASSUME_NONNULL_BEGIN

/*!
	@function SecKeychainSearchGetTypeID
	@abstract Returns the type identifier of SecKeychainSearch instances.
	@result The CFTypeID of SecKeychainSearch instances.
	@discussion This API is deprecated in 10.7. The SecKeychainSearchRef type is no longer used.
*/
CFTypeID SecKeychainSearchGetTypeID(void)
		API_DEPRECATED("SecKeychainSearch is not supported", macos(10.0, 10.7)) API_UNAVAILABLE(ios, watchos, tvos, bridgeos, macCatalyst);

/*!
	@function SecKeychainSearchCreateFromAttributes
	@abstract Creates a search reference matching a list of zero or more specified attributes in the specified keychain.
	@param keychainOrArray An reference to an array of keychains to search, a single keychain or NULL to search the user's default keychain search list.
	@param itemClass The keychain item class.
	@param attrList A pointer to a list of zero or more keychain attribute records to match.  Pass NULL to match any keychain attribute.
	@param searchRef On return, a pointer to the current search reference. You are responsible for calling the CFRelease function to release this reference when finished with it.
	@result A result code.  See "Security Error Codes" (SecBase.h).
	@discussion This function is deprecated in Mac OS X 10.7 and later; to find keychain items which match specified attributes, please use the SecItemCopyMatching API (see SecItem.h).
*/
OSStatus SecKeychainSearchCreateFromAttributes(CFTypeRef __nullable keychainOrArray, SecItemClass itemClass, const SecKeychainAttributeList * __nullable attrList, SecKeychainSearchRef * __nonnull CF_RETURNS_RETAINED searchRef)
		API_DEPRECATED("SecKeychainSearch is not supported", macos(10.0, 10.7)) API_UNAVAILABLE(ios, watchos, tvos, bridgeos, macCatalyst);

/*!
	@function SecKeychainSearchCopyNext
	@abstract Finds the next keychain item matching the given search criteria.
	@param searchRef A reference to the current search criteria.  The search reference is created in the SecKeychainSearchCreateFromAttributes function and must be released by calling the CFRelease function when you are done with it.
	@param itemRef On return, a pointer to a keychain item reference of the next matching keychain item, if any.  	
	@result A result code.  When there are no more items that match the parameters specified to SecPolicySearchCreate, errSecItemNotFound is returned. See "Security Error Codes" (SecBase.h).
	@discussion This function is deprecated in Mac OS X 10.7 and later; to find keychain items which match specified attributes, please use the SecItemCopyMatching API (see SecItem.h).
*/
OSStatus SecKeychainSearchCopyNext(SecKeychainSearchRef searchRef, SecKeychainItemRef * __nonnull CF_RETURNS_RETAINED itemRef)
		API_DEPRECATED("SecKeychainSearch is not supported", macos(10.0, 10.7)) API_UNAVAILABLE(ios, watchos, tvos, bridgeos, macCatalyst);

CF_ASSUME_NONNULL_END

#if defined(__cplusplus)
}
#endif

#endif /* !_SECURITY_SECKEYCHAINSEARCH_H_ */
