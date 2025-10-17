/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 25, 2022.
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
#include <TargetConditionals.h>
#if TARGET_OS_IPHONE && !TARGET_OS_SIMULATOR

#include "SecurityCommands.h"

#include "security.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <CoreFoundation/CFNumber.h>
#include <CoreFoundation/CFString.h>
#include <Security/SecCertificatePriv.h>
#include <Security/SecTrustStore.h>

#include "SecurityTool/sharedTool/readline.h"
#include "SecurityTool/sharedTool/tool_errors.h"
#include <utilities/SecCFWrappers.h>


static int
do_add_certificates(const char *keychainName, bool trustSettings,
	int argc, char * const *argv)
{
	int ix, result = 0;
	OSStatus status;

	CFMutableDictionaryRef attributes =
		CFDictionaryCreateMutable(NULL, 0, NULL, NULL);
	CFDictionarySetValue(attributes, kSecClass, kSecClassCertificate);

	for (ix = 0; ix < argc; ++ix) {
        CFDataRef data = copyFileContents(argv[ix]);
        if (data) {
            SecCertificateRef cert = SecCertificateCreateWithData(
                kCFAllocatorDefault, data);
            if (!cert) {
                cert = SecCertificateCreateWithPEM(kCFAllocatorDefault, data);
            }
            CFRelease(data);
            if (cert) {
				if (trustSettings) {
					SecTrustStoreSetTrustSettings(
						SecTrustStoreForDomain(kSecTrustStoreDomainUser),
						cert, NULL);
                    CFReleaseNull(cert);
				} else {
					CFDictionarySetValue(attributes, kSecValueRef, cert);
					status = SecItemAdd(attributes, NULL);
					CFRelease(cert);
					if (status) {
						fprintf(stderr, "file %s: SecItemAdd %s",
							argv[ix], sec_errstr(status));
						result = 1;
					}
				}
            } else {
                result = 1;
                fprintf(stderr, "file %s: does not contain a valid certificate",
                    argv[ix]);
            }
        } else {
            result = 1;
        }
    }

    CFRelease(attributes);

	return result;
}


int
keychain_add_certificates(int argc, char * const *argv)
{
	int ch, result = 0;
	const char *keychainName = NULL;
	bool trustSettings = false;
	while ((ch = getopt(argc, argv, "hk:t")) != -1)
	{
		switch  (ch)
		{
        case 'k':
            keychainName = optarg;
			if (*keychainName == '\0')
				return SHOW_USAGE_MESSAGE;
            break;
        case 't':
            trustSettings = true;
            break;
		case '?':
		default:
			return SHOW_USAGE_MESSAGE;
		}
	}

	argc -= optind;
	argv += optind;

	if (argc == 0)
		return SHOW_USAGE_MESSAGE;

	result = do_add_certificates(keychainName, trustSettings, argc, argv);

	return result;
}

#endif // TARGET_OS_IPHONE && !TARGET_OS_SIMULATOR
