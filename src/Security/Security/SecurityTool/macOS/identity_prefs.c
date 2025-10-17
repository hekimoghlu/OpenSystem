/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 25, 2023.
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
#include "identity_prefs.h"
#include "identity_find.h"
#include "keychain_utilities.h"
#include "security_tool.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <Security/cssmtype.h>
#include <Security/SecCertificate.h>
#include <Security/SecIdentity.h>
#include <Security/SecIdentitySearch.h>
#include <Security/SecItem.h>
#include <CommonCrypto/CommonDigest.h>

// SecCertificateInferLabel, SecDigestGetData
#include <Security/SecCertificatePriv.h>

static int
do_set_identity_preference(CFTypeRef keychainOrArray,
	const char *identity,
	const char *service,
	CSSM_KEYUSE keyUsage,
	const char *hash)
{
	int result = 0;
	CFStringRef serviceRef = NULL;
	SecIdentityRef identityRef = NULL;

	// must have a service name
	if (!service) {
		return SHOW_USAGE_MESSAGE;
	}

	// find identity (if specified by name or hash)
	if (identity || hash) {
		identityRef = CopyMatchingIdentity(keychainOrArray, identity, hash, keyUsage);
		if (!identityRef) {
			sec_error("No matching identity found for \"%s\"", (hash) ? hash : identity);
			result = 1;
			goto cleanup;
		}
	}

	// set the identity preference
	serviceRef = CFStringCreateWithCString(NULL, service, kCFStringEncodingUTF8);
	result = SecIdentitySetPreference(identityRef, serviceRef, keyUsage);

cleanup:
	if (identityRef)
		CFRelease(identityRef);
	if (serviceRef)
		CFRelease(serviceRef);

	return result;
}

typedef struct {
    int i;
    const char *name;
} ctk_print_context;

OSStatus ctk_dump_item(CFTypeRef item, ctk_print_context *ctx);

static int
do_get_identity_preference(const char *service,
	CSSM_KEYUSE keyUsage,
	Boolean printName,
	Boolean printHash,
	Boolean pemFormat)
{
	int result = 0;
	if (!service) {
		return SHOW_USAGE_MESSAGE;
	}
	CFStringRef serviceRef = CFStringCreateWithCString(NULL, service, kCFStringEncodingUTF8);
	SecCertificateRef certRef = NULL;
	SecIdentityRef identityRef = NULL;
	CSSM_DATA certData = { 0, NULL };

	result = SecIdentityCopyPreference(serviceRef, keyUsage, NULL, &identityRef);
	if (result) {
		sec_perror("SecIdentityCopyPreference", result);
		goto cleanup;
	}
	result = SecIdentityCopyCertificate(identityRef, &certRef);
	if (result) {
		sec_perror("SecIdentityCopyCertificate", result);
		goto cleanup;
	}
	result = SecCertificateGetData(certRef, &certData);
	if (result) {
		sec_perror("SecCertificateGetData", result);
		goto cleanup;
	}

	if (printName) {
		char *nameBuf = NULL;
		CFStringRef nameRef = NULL;
		(void)SecCertificateCopyCommonName(certRef, &nameRef);
		CFIndex nameLen = (nameRef) ? CFStringGetLength(nameRef) : 0;
		if (nameLen > 0) {
			CFIndex bufLen = 1 + CFStringGetMaximumSizeForEncoding(nameLen, kCFStringEncodingUTF8);
			nameBuf = (char *)malloc(bufLen);
			if (!CFStringGetCString(nameRef, nameBuf, bufLen-1, kCFStringEncodingUTF8))
				nameBuf[0]=0;
		}
		fprintf(stdout, "common name: \"%s\"\n", (nameBuf && nameBuf[0] != 0) ? nameBuf : "<NULL>");
		if (nameBuf)
			free(nameBuf);
		safe_CFRelease(&nameRef);
	}

	if (printHash) {
		uint8_t digest[CC_SHA256_DIGEST_LENGTH];
		unsigned int i;
		if (certData.Data != NULL && certData.Length > 0)  {
			// print SHA-256 hash value
			CC_SHA256(certData.Data, (CC_LONG)certData.Length, digest);
			fprintf(stdout, "SHA-256 hash: ");
			for (i=0; i<CC_SHA256_DIGEST_LENGTH; i++) {
				fprintf(stdout, "%02X", ((unsigned char *)digest)[i]);
			}
			fprintf(stdout, "\n");
			// print SHA-1 hash value for compatibility
			CC_SHA1(certData.Data, (CC_LONG)certData.Length, digest);
			fprintf(stdout, "SHA-1 hash: ");
			for (i=0; i<CC_SHA1_DIGEST_LENGTH; i++) {
				fprintf(stdout, "%02X", ((unsigned char *)digest)[i]);
			}
			fprintf(stdout, "\n");
		}
	}

	if (pemFormat)
	{
		CSSM_DATA certData = { 0, NULL };
		result = SecCertificateGetData(certRef, &certData);
		if (result) {
			sec_perror("SecCertificateGetData", result);
			goto cleanup;
		}

		print_buffer_pem(stdout, "CERTIFICATE", certData.Length, certData.Data);
	}
	else
	{
        CFTypeRef keys[] = { kSecValueRef, kSecReturnAttributes };
        CFTypeRef values[] = { certRef, kCFBooleanTrue };
        CFDictionaryRef query = CFDictionaryCreate(kCFAllocatorDefault, keys, values, sizeof(keys) / sizeof(CFTypeRef), &kCFTypeDictionaryKeyCallBacks, &kCFTypeDictionaryValueCallBacks);
        CFDictionaryRef attributes = NULL;
        if (SecItemCopyMatching(query, (void *)&attributes) == errSecSuccess && CFDictionaryContainsKey(attributes, kSecAttrTokenID)) {
            ctk_print_context ctx = { 1, "certificate" };
            ctk_dump_item(attributes, &ctx);
        } else
            print_keychain_item_attributes(stdout, (SecKeychainItemRef)certRef, FALSE, FALSE, FALSE, FALSE);

        CFRelease(query);
        if(attributes)
            CFRelease(attributes);
	}

cleanup:
	safe_CFRelease(&serviceRef);
	safe_CFRelease(&certRef);
	safe_CFRelease(&identityRef);

	return result;
}

int
set_identity_preference(int argc, char * const *argv)
{
	int ch, result = 0;
	char *identity = NULL, *service = NULL, *hash = NULL;
	CSSM_KEYUSE keyUsage = 0;
	CFTypeRef keychainOrArray = NULL;

	/*
	 *	"    -n  Specify no identity (clears existing preference for service)\n"
	 *	"    -c  Specify identity by common name of the certificate\n"
	 *	"    -s  Specify service (URI, email address, DNS host, or other name)\n"
	 *  "        for which this identity is to be preferred\n"
	 *	"    -u  Specify key usage (optional)\n"
	 *	"    -Z  Specify identity by SHA-256 (or SHA-1) hash of certificate (optional)\n"
	 */

	while ((ch = getopt(argc, argv, "hnc:s:u:Z:")) != -1)
	{
		switch  (ch)
		{
			case 'n':
				identity = NULL;
				break;
			case 'c':
				identity = optarg;
				break;
			case 's':
				service = optarg;
				break;
			case 'u':
				keyUsage = atoi(optarg);
				break;
			case 'Z':
				hash = optarg;
				break;
			case '?':
			default:
				result = SHOW_USAGE_MESSAGE;
				goto cleanup;
		}
	}

	argc -= optind;
	argv += optind;

    keychainOrArray = keychain_create_array(argc, argv);

	result = do_set_identity_preference(keychainOrArray, identity, service, keyUsage, hash);

cleanup:
	safe_CFRelease(&keychainOrArray);

	return result;
}

int
get_identity_preference(int argc, char * const *argv)
{
	int ch, result = 0;
	char *service = NULL;
	Boolean printName = FALSE, printHash = FALSE, pemFormat = FALSE;
	CSSM_KEYUSE keyUsage = 0;

	/*
	 *	"    -s  Specify service (URI, email address, DNS host, or other name)\n"
	 *	"    -u  Specify key usage (optional)\n"
	 *  "    -p  Output identity certificate in pem format\n"
	 *	"    -c  Print common name of the preferred identity certificate (optional)\n"
	 *	"    -Z  Print SHA-256 (or SHA-1) hash of the preferred identity certificate (optional)\n"
	 */

	while ((ch = getopt(argc, argv, "hs:u:pcZ")) != -1)
	{
		switch  (ch)
		{
			case 'c':
				printName = TRUE;
				break;
			case 'p':
				pemFormat = TRUE;
				break;
			case 's':
				service = optarg;
				break;
			case 'u':
				keyUsage = atoi(optarg);
				break;
			case 'Z':
				printHash = TRUE;
				break;
			case '?':
			default:
				result = 2; /* @@@ Return 2 triggers usage message. */
				goto cleanup;
		}
	}

	result = do_get_identity_preference(service, keyUsage, printName, printHash, pemFormat);

cleanup:

	return result;
}

