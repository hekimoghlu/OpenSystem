/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 20, 2022.
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
	@header SecPolicySearch
	The functions provided in SecPolicySearch implement a query for SecPolicy objects.
*/

#ifndef _SECURITY_SECPOLICYSEARCH_H_
#define _SECURITY_SECPOLICYSEARCH_H_

#include <Security/SecBase.h>
#include <Security/cssmtype.h>


#if defined(__cplusplus)
extern "C" {
#endif

CF_ASSUME_NONNULL_BEGIN

/*!
	@typedef SecPolicySearchRef
	@abstract A reference to an opaque policy search structure.
*/
typedef struct CF_BRIDGED_TYPE(id) OpaquePolicySearchRef *SecPolicySearchRef;

/*!
	@function SecPolicySearchGetTypeID
	@abstract Returns the type identifier of SecPolicySearch instances.
	@result The CFTypeID of SecPolicySearch instances.
	@discussion This API is deprecated in 10.7. The SecPolicySearchRef type is no longer used.
*/
CFTypeID SecPolicySearchGetTypeID(void)
		DEPRECATED_IN_MAC_OS_X_VERSION_10_7_AND_LATER;

/*!
	@function SecPolicySearchCreate
	@abstract Creates a search reference for finding a policy by specifying its object identifier.
	@param certType The type of certificates a policy uses.
    @param policyOID A pointer to a BER-encoded policy object identifier that uniquely specifies the policy.
	@param value Unused.  Pass NULL for this value.  Use SecPolicySetValue to set per policy data.
	@param searchRef On return, a pointer to a policy search reference. The policy search reference is used for subsequent calls to the SecCopyNextPolicy function to obtain the remaining trust policies. You are responsible for releasing the search reference by calling the CFRelease function when finished with it.
    @result A result code.  See "Security Error Codes" (SecBase.h).
	@discussion This function is deprecated in 10.7. To create a SecPolicyRef, use one of the SecPolicyCreate functions in SecPolicy.h.
*/
OSStatus SecPolicySearchCreate(CSSM_CERT_TYPE certType, const CSSM_OID *policyOID, const CSSM_DATA * __nullable value, SecPolicySearchRef * __nonnull CF_RETURNS_RETAINED searchRef)
		DEPRECATED_IN_MAC_OS_X_VERSION_10_7_AND_LATER;

/*!
	@function SecPolicySearchCopyNext
	@abstract Finds the next policy matching the given search criteria
	@param searchRef A reference to the current policy search criteria.	You create the policy search  reference by a calling the SecPolicySearchCreate function. You are responsible for releasing the policy by calling the CFRelease function when finished with it.
	@param policyRef On return, a pointer to a policy reference.
	@result A result code.  When there are no more policies that match the parameters specified to SecPolicySearchCreate, errSecPolicyNotFound is returned. See "Security Error Codes" (SecBase.h).
	@discussion This function is deprecated in 10.7. To create a SecPolicyRef, use one of the SecPolicyCreate functions in SecPolicy.h.
*/
OSStatus SecPolicySearchCopyNext(SecPolicySearchRef searchRef, SecPolicyRef * __nonnull CF_RETURNS_RETAINED policyRef)
		DEPRECATED_IN_MAC_OS_X_VERSION_10_7_AND_LATER;

CF_ASSUME_NONNULL_END

#if defined(__cplusplus)
}
#endif

#endif /* !_SECURITY_SECPOLICY_H_ */
