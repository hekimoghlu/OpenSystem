/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 22, 2024.
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
	@header SecIdentitySearch
	The functions provided in SecIdentitySearch implement a query for SecIdentity objects.
*/

#ifndef _SECURITY_SECIDENTITYSEARCH_H_
#define _SECURITY_SECIDENTITYSEARCH_H_

#include <Security/SecBase.h>
#include <Security/cssmtype.h>
#include <CoreFoundation/CFArray.h>
#include <CoreFoundation/CFDictionary.h>
#include <CoreFoundation/CFString.h>
#include <AvailabilityMacros.h>


#if defined(__cplusplus)
extern "C" {
#endif

CF_ASSUME_NONNULL_BEGIN

/*!
    @typedef SecIdentitySearchRef
    @abstract Contains information about an identity search.
*/
typedef struct CF_BRIDGED_TYPE(id) OpaqueSecIdentitySearchRef *SecIdentitySearchRef;

/*!
	@function SecIdentitySearchGetTypeID
	@abstract Returns the type identifier of SecIdentitySearch instances.
	@result The CFTypeID of SecIdentitySearch instances.
	@discussion This API is deprecated in 10.7. The SecIdentitySearchRef type is no longer used.
*/
CFTypeID SecIdentitySearchGetTypeID(void)
		DEPRECATED_IN_MAC_OS_X_VERSION_10_7_AND_LATER;

/*!
	@function SecIdentitySearchCreate
	@abstract Creates a search reference for finding identities.
    @param keychainOrArray An reference to an array of keychains to search, a single keychain, or NULL to search the user's default keychain search list.
	@param keyUsage A CSSM_KEYUSE value, as defined in cssmtype.h. This value narrows the search to return only those identities which match the specified key usage. Pass a value of 0 to ignore key usage and return all available identities. Note that passing CSSM_KEYUSE_ANY limits the results to only those identities that can be used for every operation.
    @param searchRef On return, an identity search reference. You must release the identity search reference by calling the CFRelease function.
    @result A result code.  See "Security Error Codes" (SecBase.h).
	@discussion You can set values for key usage, and one or more keychains, to control the search for identities. You can use the returned search reference to obtain the remaining identities in subsequent calls to the SecIentitySearchCopyNext function. You must release the identity search reference by calling the CFRelease function.
	This function is deprecated in Mac OS X 10.7 and later; to find identities which match a given key usage or other attributes, please use the SecItemCopyMatching API (see SecItem.h).
*/
OSStatus SecIdentitySearchCreate(CFTypeRef __nullable keychainOrArray, CSSM_KEYUSE keyUsage, SecIdentitySearchRef * __nullable CF_RETURNS_RETAINED searchRef)
		DEPRECATED_IN_MAC_OS_X_VERSION_10_7_AND_LATER;

/*!
	@function SecIdentitySearchCopyNext
    @abstract Finds the next identity matching the given search criteria, as previously specified by a call to SecIdentitySearchCreate or SecIdentitySearchCreateWithAttributes.
	@param searchRef A reference to the current identity search. You create the identity search reference by calling either SecIdentitySearchCreate or SecIdentitySearchCreateWithAttributes.
	@param identity On return, an identity reference for the next found identity, if any. You must call the CFRelease function when finished with the identity reference.
	@result A result code. When there are no more identities found that match the search criteria, errSecItemNotFound is returned. See "Security Error Codes" (SecBase.h).
	@discussion This function is deprecated in Mac OS X 10.7 and later; to find identities which match specified attributes, please use the SecItemCopyMatching API (see SecItem.h).
*/
OSStatus SecIdentitySearchCopyNext(SecIdentitySearchRef searchRef, SecIdentityRef * __nullable CF_RETURNS_RETAINED identity)
		DEPRECATED_IN_MAC_OS_X_VERSION_10_7_AND_LATER;

CF_ASSUME_NONNULL_END

#if defined(__cplusplus)
}
#endif

#endif /* !_SECURITY_SECIDENTITYSEARCH_H_ */
