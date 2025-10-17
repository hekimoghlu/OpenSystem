/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 19, 2025.
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
#import <Security/Security.h>

#include "keychain_regressions.h"
#include "kc-helpers.h"

//	Tests the ability of SecItemCopyMatching to return a persistent ref for either a
//	SecIdentityRef or a SecCertificateRef which happens to be part of an identity,
//	then reconstitute the appropriate type of ref from the persistent reference.
//

#include <CoreFoundation/CoreFoundation.h>
#include <CoreServices/CoreServices.h>
#include <Security/Security.h>

#include <stdlib.h>
#include <string.h>
#include <syslog.h>
#include <unistd.h>
#include <time.h>
#include <sys/param.h>
#include "kc-identity-helpers.h"

#define MAXNAMELEN MAXPATHLEN
#define MAXITEMS INT32_MAX

static CFDataRef
copyPersistentReferenceForItem(CFTypeRef item)
{
	// Given either a SecIdentityRef or SecCertificateRef item reference,
	// return a persistent reference. Caller must release the reference.

	OSStatus status;
	CFDataRef persistentRef = NULL;
	CFDictionaryRef query = NULL;

	const void *keys[] = { kSecReturnPersistentRef, kSecValueRef };
	const void *values[] = { kCFBooleanTrue, item };

	query = CFDictionaryCreate(NULL, keys, values,
		(sizeof(keys) / sizeof(*keys)), NULL, NULL);
	status = SecItemCopyMatching(query, (CFTypeRef *)&persistentRef);
    ok_status(status, "%s: SecItemCopyMatching (copyPersistentReferenceForItem)", testName);
	CFRelease(query);
	return persistentRef;
}

static CFTypeRef
copyItemForPersistentReference(CFDataRef persistentRef)
{
	// Given a persistent reference, reconstitute it into an item
	// reference. Depending on whether the persistent reference was
	// originally made to a SecIdentityRef or SecCertificateRef, this
	// should return the same item type as the original.
	// Caller must release the reference.

	OSStatus status;
	CFTypeRef itemRef = NULL;
	CFDictionaryRef query = NULL;
	const void *keys[] = { kSecReturnRef, kSecValuePersistentRef };
	const void *values[] = { kCFBooleanTrue, persistentRef };

	query = CFDictionaryCreate(NULL, keys, values,
		(sizeof(keys) / sizeof(*keys)), NULL, NULL);
	status = SecItemCopyMatching(query, &itemRef);
    ok_status(status, "%s: SecItemCopyMatching (copyItemForPersistentReference)", testName);
	CFRelease(query);
	return itemRef;
}

static void
testIdentityPersistence(SecKeychainRef kc)
{
    startTest(__FUNCTION__);

	// Step 1: get a SecIdentityRef
	SecIdentityRef identity = copyFirstIdentity(kc);
    is(CFGetTypeID(identity), SecIdentityGetTypeID(), "%s: retrieved identity is an identity", testName);

	// Step 2: make a persistent reference for it
	CFDataRef data = copyPersistentReferenceForItem((CFTypeRef)identity);

	// Step 3: reconstitute the persistent reference
	SecIdentityRef identity2 = (SecIdentityRef) copyItemForPersistentReference(data);
    CFReleaseNull(data);

    ok(identity2, "%s: retrieved an identity", testName);
    if(identity2) {
        is(CFGetTypeID(identity2), SecIdentityGetTypeID(), "%s: retrieved identity is an identity", testName);
    } else {
        fail("%s: no identity to test", testName);
    }
    eq_cf(identity2, identity, "%s: identities are equal", testName);

    CFReleaseNull(identity);
    CFReleaseNull(identity2);
}

static void
testCertificatePersistence(SecKeychainRef kc)
{
    startTest(__FUNCTION__);

	// Step 1: get a SecIdentityRef
	SecIdentityRef identity = copyFirstIdentity(kc);

	// Step 2: get a SecCertificateRef from it
	SecCertificateRef cert = NULL, cert2 = NULL;
	OSStatus status = SecIdentityCopyCertificate(identity, &cert);
    ok_status(status, "%s: SecIdentityCopyCertificate", testName);
    ok(cert, "%s: No certificate returned from SecIdentityCopyCertificate", testName);
    CFReleaseNull(identity);

	// Step 3: make a persistent reference for it
	CFDataRef data = (CFDataRef) copyPersistentReferenceForItem(cert);

	// Step 4: reconstitute the persistent reference
	cert2 = (SecCertificateRef) copyItemForPersistentReference(data);

    ok(cert2, "%s: retrieved a certificate", testName);
    if(cert2) {
        is(CFGetTypeID(cert2), SecCertificateGetTypeID(), "%s: returned value is a certificate", testName);
    } else {
        fail("%s: no certificate to test", testName);
    }

    eq_cf(cert2, cert, "%s: Certificates are equal", testName);

    CFReleaseNull(data);
    CFReleaseNull(cert);
    CFReleaseNull(cert2);
}

int kc_20_identity_persistent_refs(int argc, char *const *argv)
{
    plan_tests(18);
    initializeKeychainTests(__FUNCTION__);

    SecKeychainRef kc = getPopulatedTestKeychain();

    // You cannot reconsitute a Persistent Reference for an identity if the keychain is not in the search list.
    addToSearchList(kc);

	testCertificatePersistence(kc);
	testIdentityPersistence(kc);

    ok_status(SecKeychainDelete(kc), "%s: SecKeychainDelete", testName);
    CFReleaseNull(kc);

    deleteTestFiles();
    return 0;
}
