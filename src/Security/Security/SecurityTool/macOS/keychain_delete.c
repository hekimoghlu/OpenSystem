/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 17, 2023.
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
#include "keychain_delete.h"
#include "keychain_find.h"

#include "keychain_utilities.h"
#include "security_tool.h"
#include <unistd.h>
#include <Security/SecIdentity.h>
#include <Security/SecKeychain.h>
#include <Security/SecKeychainItem.h>
#include <Security/SecTrustSettings.h>

static int
do_delete(CFTypeRef keychainOrArray)
{
	/* @@@ SecKeychainDelete should really take a CFTypeRef argument. */
	OSStatus result = SecKeychainDelete((SecKeychainRef)keychainOrArray);
	if (result)
	{
		/* @@@ Add printing of keychainOrArray. */
		sec_perror("SecKeychainDelete", result);
	}

	return result;
}

static int
do_delete_certificate(CFTypeRef keychainOrArray, const char *name, const char *hash,
                      Boolean deleteTrust, Boolean deleteIdentity)
{
    OSStatus result = noErr;
    SecKeychainItemRef itemToDelete = NULL;
    if (!name && !hash) {
        return SHOW_USAGE_MESSAGE;
    }

    itemToDelete = find_unique_certificate(keychainOrArray, name, hash);
    if (itemToDelete) {
        OSStatus status = noErr;
        if (deleteTrust) {
            status = SecTrustSettingsRemoveTrustSettings((SecCertificateRef)itemToDelete,
                                                         kSecTrustSettingsDomainUser);
            if (status) {
                // if trust settings do not exist, it's not an error.
                if (status != errSecItemNotFound) {
                    result = status;
                    sec_perror("SecTrustSettingsRemoveTrustSettings (user)", result);
                }
            }
            if (geteuid() == 0) {
                status = SecTrustSettingsRemoveTrustSettings((SecCertificateRef)itemToDelete,
                                                             kSecTrustSettingsDomainAdmin);
                if (status) {
                    if (status != errSecItemNotFound) {
                        result = status;
                        sec_perror("SecTrustSettingsRemoveTrustSettings (admin)", result);
                    }
                }
            }
        }
        if (!result && deleteIdentity) {
            SecIdentityRef identity = NULL;
            status = SecIdentityCreateWithCertificate(keychainOrArray,
                                                      (SecCertificateRef)itemToDelete,
                                                      &identity);
            if (status) {
                // if the private key doesn't exist, and we succeed in deleting
                // the certificate, overall result will still be good.
                if (status == errSecItemNotFound) {
                    status = noErr;
                } else {
                    result = status;
                }
            } else {
                SecKeyRef keyToDelete = NULL;
                status = SecIdentityCopyPrivateKey(identity, &keyToDelete);
                if (status) {
                    result = status;
                } else {
                    result = SecKeychainItemDelete((SecKeychainItemRef)keyToDelete);
                    if (result) {
                        sec_perror("SecKeychainItemDelete", result);
                    }
                }
                safe_CFRelease(&keyToDelete);
            }
            safe_CFRelease(&identity);

            if (status) {
                fprintf(stderr, "Unable to obtain private key reference for \"%s\" (error %d)",
                        (name) ? name : (hash) ? hash : "", (int) status);
            }
        }
        if (!result) {
            result = SecKeychainItemDelete(itemToDelete);
            if (result) {
                sec_perror("SecKeychainItemDelete", result);
                goto cleanup;
            }
        }
    } else {
        result = 1;
        fprintf(stderr, "Unable to delete certificate matching \"%s\"",
                (name) ? name : (hash) ? hash : "");
    }

cleanup:
    safe_CFRelease(&itemToDelete);

    return result;
}

static int
keychain_delete_cert_common(int argc, char * const *argv, Boolean delete_identity)
{
	CFTypeRef keychainOrArray = NULL;
	char *name = NULL;
	char *hash = NULL;
	Boolean delete_trust = FALSE;
	int ch, result = 0;

	while ((ch = getopt(argc, argv, "hc:Z:t")) != -1)
	{
		switch  (ch)
		{
			case 'c':
				name = optarg;
				break;
			case 'Z':
				hash = optarg;
				break;
			case 't':
				delete_trust = TRUE;
				break;
			case '?':
			default:
				result = 2; /* @@@ Return 2 triggers usage message. */
				goto cleanup;
		}
	}

	argc -= optind;
	argv += optind;

	keychainOrArray = keychain_create_array(argc, argv);

	result = do_delete_certificate(keychainOrArray, name, hash, delete_trust, delete_identity);

cleanup:
	safe_CFRelease(&keychainOrArray);

	return result;
}

int
keychain_delete_certificate(int argc, char * const *argv)
{
    return keychain_delete_cert_common(argc, argv, FALSE);
}

int
keychain_delete_identity(int argc, char * const *argv)
{
    return keychain_delete_cert_common(argc, argv, TRUE);
}

int
keychain_delete(int argc, char * const *argv)
{
	CFTypeRef keychainOrArray = NULL;
	int ch, result = 0;

	while ((ch = getopt(argc, argv, "h")) != -1)
	{
		switch  (ch)
		{
		case '?':
		default:
			return SHOW_USAGE_MESSAGE;
		}
	}

	argc -= optind;
	argv += optind;

	keychainOrArray = keychain_create_array(argc, argv);

	result = do_delete(keychainOrArray);
	if (keychainOrArray)
		CFRelease(keychainOrArray);

	return result;
}
