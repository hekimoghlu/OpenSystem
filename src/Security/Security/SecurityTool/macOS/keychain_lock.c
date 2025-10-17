/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 29, 2022.
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
#include "keychain_lock.h"

#include "keychain_utilities.h"
#include "readline_cssm.h"
#include "security_tool.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <Security/SecKeychain.h>

static int
do_lock_all(void)
{
	OSStatus result = SecKeychainLockAll();
    if (result) {
        sec_perror("SecKeychainLockAll", result);
    }

    return result;
}

static int
do_lock(const char *keychainName)
{
	SecKeychainRef keychain = NULL;
	OSStatus result;

	if (keychainName)
	{
		keychain = keychain_open(keychainName);
		if (!keychain)
		{
			result = 1;
			goto loser;
		}
	}

	result = SecKeychainLock(keychain);
	if (result)
	{
		sec_error("SecKeychainLock %s: %s", keychainName ? keychainName : "<NULL>", sec_errstr(result));
	}

loser:
	if (keychain)
		CFRelease(keychain);

	return result;
}

int
keychain_lock(int argc, char * const *argv)
{
	char *keychainName = NULL;
	int ch, result = 0;
	Boolean lockAll = FALSE;
	while ((ch = getopt(argc, argv, "ah")) != -1)
	{
		switch  (ch)
		{
		case 'a':
			lockAll = TRUE;
			break;
		case '?':
		default:
			return SHOW_USAGE_MESSAGE;
		}
	}
	argc -= optind;
	argv += optind;

	if (argc == 1 && !lockAll)
	{
		keychainName = argv[0];
		if (*keychainName == '\0')
		{
			result = 2;
			goto loser;
		}
	}
	else if (argc != 0)
		return SHOW_USAGE_MESSAGE;

	if (lockAll)
		result = do_lock_all();
	else
		result = do_lock(keychainName);

loser:

	return result;
}
