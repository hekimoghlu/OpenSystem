/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 19, 2022.
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
#include <Security/SecKeychainSearch.h>
#include <Security/SecKeychainSearchPriv.h>
#include <Security/SecCertificatePriv.h>
#include <security_keychain/KCCursor.h>
#include <security_keychain/Certificate.h>
#include <security_keychain/Item.h>
#include <security_cdsa_utilities/Schema.h>
#include <syslog.h>
#include <os/activity.h>

#include "SecBridge.h"
#include "LegacyAPICounts.h"

CFTypeID
SecKeychainSearchGetTypeID(void)
{
	BEGIN_SECAPI
	return gTypes().KCCursorImpl.typeID;

	END_SECAPI1(_kCFRuntimeNotATypeID)
}


OSStatus
SecKeychainSearchCreateFromAttributes(CFTypeRef keychainOrArray, SecItemClass itemClass, const SecKeychainAttributeList *attrList, SecKeychainSearchRef *searchRef)
{
    BEGIN_SECAPI
    os_activity_t activity = os_activity_create("SecKeychainSearchCreateFromAttributes", OS_ACTIVITY_CURRENT, OS_ACTIVITY_FLAG_IF_NONE_PRESENT);
    os_activity_scope(activity);
    os_release(activity);

	Required(searchRef);

	StorageManager::KeychainList keychains;
	globals().storageManager.optionalSearchList(keychainOrArray, keychains);
	KCCursor cursor(keychains, itemClass, attrList);
	*searchRef = cursor->handle();

	END_SECAPI
}


OSStatus
SecKeychainSearchCreateFromAttributesExtended(CFTypeRef keychainOrArray, SecItemClass itemClass, const SecKeychainAttributeList *attrList, CSSM_DB_CONJUNCTIVE dbConjunctive, CSSM_DB_OPERATOR dbOperator, SecKeychainSearchRef *searchRef)
{
    BEGIN_SECAPI
    os_activity_t activity = os_activity_create("SecKeychainSearchCreateFromAttributesExtended", OS_ACTIVITY_CURRENT, OS_ACTIVITY_FLAG_IF_NONE_PRESENT);
    os_activity_scope(activity);
    os_release(activity);

	Required(searchRef); // Make sure that searchRef is an invalid SearchRef

	StorageManager::KeychainList keychains;
	globals().storageManager.optionalSearchList(keychainOrArray, keychains);
	KCCursor cursor(keychains, itemClass, attrList, dbConjunctive, dbOperator);

	*searchRef = cursor->handle();

	END_SECAPI
}



OSStatus
SecKeychainSearchCopyNext(SecKeychainSearchRef searchRef, SecKeychainItemRef *itemRef)
{
	BEGIN_SECAPI
    os_activity_t activity = os_activity_create("SecKeychainSearchCopyNext", OS_ACTIVITY_CURRENT, OS_ACTIVITY_FLAG_IF_NONE_PRESENT);
    os_activity_scope(activity);
    os_release(activity);

	RequiredParam(itemRef);
	Item item;
	KCCursorImpl *itemCursor = KCCursorImpl::required(searchRef);
	if (!itemCursor->next(item))
		return errSecItemNotFound;

	*itemRef=item->handle();

	bool itemChecked = false;
	do {
		/* see if we should convert outgoing item to a unified SecCertificateRef */
		SecItemClass tmpItemClass = Schema::itemClassFor(item->recordType());
		if (tmpItemClass == kSecCertificateItemClass) {
			SecPointer<Certificate> certificate(static_cast<Certificate *>(&*item));
			CssmData certData = certificate->data();
			CFDataRef data = NULL;
			if (certData.Data && certData.Length) {
				data = CFDataCreate(NULL, certData.Data, certData.Length);
			}
			if (!data) {
				/* zero-length or otherwise bad cert data; skip to next item */
				if (*itemRef) {
					CFRelease(*itemRef);
					*itemRef = NULL;
				}
				if (!itemCursor->next(item))
					return errSecItemNotFound;
				*itemRef=item->handle();
				continue;
			}
			SecKeychainItemRef tmpRef = *itemRef;
			*itemRef = (SecKeychainItemRef) SecCertificateCreateWithKeychainItem(NULL, data, tmpRef);
			if (data)
				CFRelease(data);
			if (tmpRef)
				CFRelease(tmpRef);
			if (NULL == *itemRef) {
				/* unable to create unified certificate item; skip to next item */
				if (!itemCursor->next(item))
					return errSecItemNotFound;
				*itemRef=item->handle();
				continue;
			}
			itemChecked = true;
		}
		else {
			itemChecked = true;
		}
	} while (!itemChecked);

	if (NULL == *itemRef) {
		/* never permit a NULL item reference to be returned without an error result */
		return errSecItemNotFound;
	}

	END_SECAPI
}
