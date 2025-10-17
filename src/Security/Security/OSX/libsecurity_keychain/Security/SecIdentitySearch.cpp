/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 17, 2023.
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
#include <Security/SecIdentitySearch.h>
#include <Security/SecIdentitySearchPriv.h>
#include <Security/SecPolicyPriv.h>
#include <security_keychain/IdentityCursor.h>
#include <security_keychain/Identity.h>
#include <os/activity.h>

#include "SecBridge.h"
#include "LegacyAPICounts.h"

CFTypeID
SecIdentitySearchGetTypeID(void)
{
	BEGIN_SECAPI

	return gTypes().IdentityCursor.typeID;

	END_SECAPI1(_kCFRuntimeNotATypeID)
}


OSStatus
SecIdentitySearchCreate(
	CFTypeRef keychainOrArray,
	CSSM_KEYUSE keyUsage,
	SecIdentitySearchRef *searchRef)
{
    BEGIN_SECAPI
    os_activity_t activity = os_activity_create("SecIdentitySearchCreate", OS_ACTIVITY_CURRENT, OS_ACTIVITY_FLAG_IF_NONE_PRESENT);
    os_activity_scope(activity);
    os_release(activity);

	Required(searchRef);

	StorageManager::KeychainList keychains;
	globals().storageManager.optionalSearchList(keychainOrArray, keychains);
	SecPointer<IdentityCursor> identityCursor(new IdentityCursor (keychains, keyUsage));
	*searchRef = identityCursor->handle();

	END_SECAPI
}

OSStatus SecIdentitySearchCreateWithAttributes(
    CFDictionaryRef attributes,
    SecIdentitySearchRef* searchRef)
{
    BEGIN_SECAPI
    os_activity_t activity = os_activity_create("SecIdentitySearchCreateWithAttributes", OS_ACTIVITY_CURRENT, OS_ACTIVITY_FLAG_IF_NONE_PRESENT);
    os_activity_scope(activity);
    os_release(activity);

    //
    // %%%TBI This function needs a new form of IdentityCursor that takes
    // the supplied attributes as input.
    //
	Required(searchRef);
	StorageManager::KeychainList keychains;
	globals().storageManager.getSearchList(keychains);
	SecPointer<IdentityCursor> identityCursor(new IdentityCursor (keychains, 0));
	*searchRef = identityCursor->handle();

    END_SECAPI
}

OSStatus SecIdentitySearchCreateWithPolicy(
    SecPolicyRef policy,
    CFStringRef idString,
    CSSM_KEYUSE keyUsage,
    CFTypeRef keychainOrArray,
    Boolean returnOnlyValidIdentities,
    SecIdentitySearchRef* searchRef)
{
    BEGIN_SECAPI
    os_activity_t activity = os_activity_create("SecIdentitySearchCreateWithPolicy", OS_ACTIVITY_CURRENT, OS_ACTIVITY_FLAG_IF_NONE_PRESENT);
    os_activity_scope(activity);
    os_release(activity);

	Required(searchRef);

	StorageManager::KeychainList keychains;
	globals().storageManager.optionalSearchList(keychainOrArray, keychains);
	CFRef<SecPolicyRef> policyRef = SecPolicyCreateItemImplInstance(policy);
	SecPointer<IdentityCursorPolicyAndID> identityCursor(new IdentityCursorPolicyAndID (keychains, keyUsage, idString, policyRef, returnOnlyValidIdentities));

	*searchRef = identityCursor->handle();

	END_SECAPI
}

OSStatus
SecIdentitySearchCopyNext(
	SecIdentitySearchRef searchRef,
	SecIdentityRef *identityRef)
{
    BEGIN_SECAPI
    os_activity_t activity = os_activity_create("SecIdentitySearchCopyNext", OS_ACTIVITY_CURRENT, OS_ACTIVITY_FLAG_IF_NONE_PRESENT);
    os_activity_scope(activity);
    os_release(activity);

	RequiredParam(identityRef);
	SecPointer<Identity> identityPtr;
	if (!IdentityCursor::required(searchRef)->next(identityPtr))
		return errSecItemNotFound;

	*identityRef = identityPtr->handle();

    END_SECAPI
}
