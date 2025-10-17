/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 28, 2025.
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
#import <Security/SecCertificatePriv.h>

#include "keychain_regressions.h"
#include "kc-helpers.h"
#include "kc-identity-helpers.h"


//	Example of looking up a SecIdentityRef in the keychain,
//	then getting the attributes of its private key.
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

static void PrintPrivateKeyAttributes(SecKeyRef keyRef)
{
	CFMutableDictionaryRef query = CFDictionaryCreateMutable(NULL, 0,
			&kCFTypeDictionaryKeyCallBacks,
			&kCFTypeDictionaryValueCallBacks);

	/* set up the query: find specified item, return attributes */
	//CFDictionaryAddValue( query, kSecClass, kSecClassKey );
	CFDictionaryAddValue( query, kSecValueRef, keyRef );
	CFDictionaryAddValue( query, kSecReturnAttributes, kCFBooleanTrue );

	CFTypeRef result = NULL;
    OSStatus status = SecItemCopyMatching(query, &result);
    ok_status(status, "%s: SecItemCopyMatching", testName);

	if (query)
		CFRelease(query);

    if(result) {
        CFShow(result);
    }
}

static void tests(SecKeychainRef kc)
{
	SecIdentityRef identity=NULL;
	SecKeyRef privateKeyRef=NULL;
	OSStatus status;

	identity = copyFirstIdentity(kc);
	status = SecIdentityCopyPrivateKey(identity, &privateKeyRef);
    ok_status(status, "%s: SecIdentityCopyPrivateKey", testName);

	if (privateKeyRef) {
		PrintPrivateKeyAttributes(privateKeyRef);
		CFRelease(privateKeyRef);
	}
    CFReleaseNull(identity);
}

int kc_20_identity_key_attributes(int argc, char *const *argv)
{
    plan_tests(6);
    initializeKeychainTests(__FUNCTION__);

    SecKeychainRef kc = getPopulatedTestKeychain();

	tests(kc);

    ok_status(SecKeychainDelete(kc), "%s: SecKeychainDelete", testName);
    CFReleaseNull(kc);

    deleteTestFiles();
    return 0;
}
