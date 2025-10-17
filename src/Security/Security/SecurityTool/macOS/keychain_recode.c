/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 17, 2022.
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
#include "keychain_recode.h"

#include "keychain_utilities.h"
#include "readline_cssm.h"
#include "security_tool.h"

#include <pwd.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <CoreFoundation/CoreFoundation.h>
#include <Security/SecKeychain.h>

// SecKeychainCopyBlob, SecKeychainRecodeKeychain
#include <Security/SecKeychainPriv.h>


static int
do_recode(const char *keychainName1, const char *keychainName2)
{
	SecKeychainRef keychain1 = NULL, keychain2 = NULL;
	CFMutableArrayRef dbBlobArray = NULL;
	CFDataRef dbBlob = NULL, extraData = NULL;
	OSStatus result;

	if (keychainName1)
	{
		keychain1 = keychain_open(keychainName1);
		if (!keychain1)
		{
			result = 1;
			goto loser;
		}
	}

	keychain2 = keychain_open(keychainName2);
	if (!keychain2)
	{
		result = 1;
		goto loser;
	}

	result = SecKeychainCopyBlob(keychain2, &dbBlob);
	if (result)
	{
		sec_error("SecKeychainCopyBlob %s: %s", keychainName2,
			sec_errstr(result));
		goto loser;
	}

	extraData = CFDataCreate(NULL, NULL, 0);

	dbBlobArray = CFArrayCreateMutable(NULL, 1, &kCFTypeArrayCallBacks);
	if (dbBlobArray) {
		CFArrayAppendValue(dbBlobArray, dbBlob);
	}

#if !defined MAC_OS_X_VERSION_10_6 || MAC_OS_X_VERSION_MAX_ALLOWED < MAC_OS_X_VERSION_10_6
	result = SecKeychainRecodeKeychain(keychain1, dbBlob, extraData);
#else
	result = SecKeychainRecodeKeychain(keychain1, dbBlobArray, extraData);
#endif
	if (result)
		sec_error("SecKeychainRecodeKeychain %s, %s: %s", keychainName1,
			keychainName2, sec_errstr(result));

loser:
	if (dbBlobArray)
		CFRelease(dbBlobArray);
	if (dbBlob)
		CFRelease(dbBlob);
	if (extraData)
		CFRelease(extraData);
	if (keychain1)
		CFRelease(keychain1);
	if (keychain2)
		CFRelease(keychain2);

	return result;
}

int
keychain_recode(int argc, char * const *argv)
{
	char *keychainName1 = NULL, *keychainName2 = NULL;
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

	if (argc == 2)
	{
		keychainName1 = argv[0];
		if (*keychainName1 == '\0')
		{
			result = 2;
			goto loser;
		}

		keychainName2 = argv[1];
		if (*keychainName2 == '\0')
		{
			result = 2;
			goto loser;
		}

	}
	else
		return SHOW_USAGE_MESSAGE;

	result = do_recode(keychainName1, keychainName2);

loser:

	return result;
}
