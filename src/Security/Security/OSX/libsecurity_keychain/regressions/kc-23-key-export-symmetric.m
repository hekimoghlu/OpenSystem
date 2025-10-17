/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 3, 2023.
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

#import <Cocoa/Cocoa.h>
#import <Security/SecItemPriv.h>
#import <Security/CMSEncoder.h>
#import <Security/CMSDecoder.h>
#import <Foundation/NSData_Private.h>
#import <SecurityFoundation/SFCertificateAuthority.h>
#import <SecurityFoundation/SFCertificateAuthorityPriv.h>
#import <SecurityFoundation/CACertInfo.h>
#import <SecurityFoundation/CAKeyUsageExtension.h>
#import <SecurityFoundation/CAExtendedKeyUsageExtension.h>

#if 0
static void checkCryptoError(OSStatus status, NSString *functionName) {
	if (status != errSecSuccess) {
		NSError *underlyingError = [[NSError alloc] initWithDomain:NSOSStatusErrorDomain code:status userInfo:nil];
		NSDictionary *userInfo = [[NSDictionary alloc] initWithObjectsAndKeys:underlyingError, NSUnderlyingErrorKey, nil];

		[underlyingError release];

		CFStringRef message = SecCopyErrorMessageString(status, NULL);

		NSLog(@"%@ failed with error %d: %@: %@: %@", functionName, (int)status, underlyingError, userInfo, message);

		CFRelease(message);

		cssmPerror([functionName UTF8String], status);

		exit(EXIT_FAILURE);
	}
}
#endif

static CF_RETURNS_RETAINED SecKeyRef generateSymmetricKey(SecKeychainRef keychainRef, CFStringRef label)
{
	CFMutableDictionaryRef parameters;
	int32_t rawnum;
	CFNumberRef num;
	CFErrorRef error = NULL;
	SecKeyRef cryptokey;

	rawnum = 256;

	// Type
	parameters = CFDictionaryCreateMutable(kCFAllocatorDefault, 0, &kCFTypeDictionaryKeyCallBacks, &kCFTypeDictionaryValueCallBacks);
	CFDictionarySetValue(parameters, kSecAttrKeyType, kSecAttrKeyTypeAES);

	num = CFNumberCreate(kCFAllocatorDefault, kCFNumberSInt32Type, &rawnum);
	CFDictionarySetValue(parameters, kSecAttrKeySizeInBits, num);
    CFReleaseNull(num);

	// Store in keychain
	CFDictionarySetValue(parameters, kSecUseKeychain, keychainRef);
	CFDictionarySetValue(parameters, kSecAttrApplicationLabel, label);
	CFDictionarySetValue(parameters, kSecAttrLabel, label);


	// Extractable and permanent
	CFDictionarySetValue(parameters, kSecAttrIsExtractable, kCFBooleanTrue);
	CFDictionarySetValue(parameters, kSecAttrIsPermanent, kCFBooleanTrue);


	cryptokey = SecKeyGenerateSymmetric(parameters, &error);
    is(error, NULL, "%s: SecKeyGenerateSymmetric: %s", testName, (error) ? CFStringGetCStringPtr(CFErrorCopyDescription(error), kCFStringEncodingUTF8) : "no error");
	if (error) {
		return NULL;
	}

	return cryptokey;
}

int kc_23_key_export_symmetric(int argc, char *const *argv)
{
    plan_tests(6);
    initializeKeychainTests(__FUNCTION__);
    
    SecKeychainRef kc = getPopulatedTestKeychain();

	SecKeyRef cryptokey;
	OSStatus status;

	CFStringRef label = (__bridge_retained CFStringRef)([NSString stringWithFormat:@"Symmetric Cryptotest %ld %d", (long)time(NULL), arc4random(), nil]);
	cryptokey = generateSymmetricKey(kc, label);
    CFReleaseNull(label);

	// Using SecItemExport
	CFMutableArrayRef keyUsage = CFArrayCreateMutable(kCFAllocatorDefault, 0, &kCFTypeArrayCallBacks);
	CFArrayAppendValue(keyUsage, kSecAttrCanEncrypt);
	CFArrayAppendValue(keyUsage, kSecAttrCanDecrypt);
	CFMutableArrayRef keyAttributes = CFArrayCreateMutable(kCFAllocatorDefault, 0, &kCFTypeArrayCallBacks);
	SecItemImportExportKeyParameters exportParams;
	exportParams.version = SEC_KEY_IMPORT_EXPORT_PARAMS_VERSION;
	exportParams.flags = 0;
	exportParams.passphrase = NULL;
	exportParams.alertTitle = NULL;
	exportParams.alertPrompt = NULL;
	exportParams.accessRef = NULL;
	exportParams.keyUsage = keyUsage;
	exportParams.keyAttributes = keyAttributes;
	CFDataRef exportedKey2;
	status = SecItemExport(cryptokey, kSecFormatRawKey, 0, &exportParams, (CFDataRef *)&exportedKey2);
    ok_status(status, "%s: SecItemExport", testName);

    CFReleaseNull(cryptokey);

    is(CFDataGetLength(exportedKey2), 32, "%s: wrong AES-256 key size", testName);

    CFRelease(exportedKey2);

    ok_status(SecKeychainDelete(kc), "%s: SecKeychainDelete", testName);
    CFReleaseNull(kc);

    deleteTestFiles();
	return 0;
}
