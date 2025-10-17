/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 18, 2022.
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
#include <Security/SecPolicySearch.h>
#include <Security/SecPolicyPriv.h>
#include <security_keychain/PolicyCursor.h>
#include <security_keychain/Policies.h>
#include "SecBridge.h"

//
// CF Boilerplate
CFTypeID
SecPolicySearchGetTypeID(void)
{
	BEGIN_SECAPI
	return gTypes().PolicyCursor.typeID;

	END_SECAPI1(_kCFRuntimeNotATypeID)
}


OSStatus
SecPolicySearchCreate(
            CSSM_CERT_TYPE certType,
			const CSSM_OID* oid,
            const CSSM_DATA* value,
			SecPolicySearchRef* searchRef)
{
    BEGIN_SECAPI
	Required(searchRef);	// preflight
    PolicyCursor* pc = new PolicyCursor(oid, value);
    if (pc == NULL)
    {
        return errSecPolicyNotFound;
    }

	SecPointer<PolicyCursor> cursor(pc);
	*searchRef = cursor->handle();
	END_SECAPI
}


OSStatus
SecPolicySearchCopyNext(
            SecPolicySearchRef searchRef,
            SecPolicyRef* policyRef)
{
	BEGIN_SECAPI
	RequiredParam(policyRef);
	SecPointer<Policy> policy;

	/* bridge to support API functionality */
	CFStringRef oidStr = NULL;
	PolicyCursor *policyCursor = PolicyCursor::required(searchRef);
	do {
		if (!policyCursor->next(policy))
			return errSecPolicyNotFound;
		CssmOid oid = policy->oid();
		CFStringRef str = SecPolicyGetStringForOID(&oid);
		if (str) {
			oidStr = str;
			if (CFEqual(str, kSecPolicyAppleiChat) ||
				CFEqual(str, kSecPolicyApplePKINITClient) ||
				CFEqual(str, kSecPolicyApplePKINITServer)) {
				oidStr = NULL; /* TBD: support for PKINIT policies in unified code */
			}
			else if (policyCursor->oidProvided() == false &&
				CFEqual(str, kSecPolicyAppleRevocation)) {
				oidStr = NULL; /* filter this out unless specifically requested */
			}
		}
	}
	while (!oidStr);
	/* create and vend a unified SecPolicyRef instance */
	CFRef<CFDictionaryRef> properties = policy->properties();
	if ((*policyRef = SecPolicyCreateWithProperties(oidStr, properties)) != NULL) {
		__secapiresult = errSecSuccess;
	} else {
		__secapiresult = errSecPolicyNotFound;
	}

	END_SECAPI
}
