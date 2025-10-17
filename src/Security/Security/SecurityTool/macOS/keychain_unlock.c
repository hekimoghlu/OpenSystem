/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 12, 2022.
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
#include "keychain_unlock.h"
#include "readline_cssm.h"
#include "keychain_utilities.h"
#include "security_tool.h"

#include <pwd.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static int
do_unlock(const char *keychainName, char *password, Boolean use_password)
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

	result = SecKeychainUnlock(keychain, password ? (UInt32) strlen(password) : 0, password, use_password);
	if (result)
	{
		sec_error("SecKeychainUnlock %s: %s", keychainName ? keychainName : "<NULL>", sec_errstr(result));
	}

loser:
	if (keychain)
		CFRelease(keychain);

	return result;
}

int
keychain_unlock(int argc, char * const *argv)
{
	int zero_password = 0;
	char *password = NULL;
	int ch, result = 0;
	Boolean use_password = TRUE;
	const char *keychainName = NULL;

	while ((ch = getopt(argc, argv, "hp:u")) != -1)
	{
		switch  (ch)
		{
		case 'p':
			password = optarg;
			break;
        case 'u':
            use_password = FALSE;
			break;
		case '?':
		default:
			return SHOW_USAGE_MESSAGE;
		}
	}

	argc -= optind;
	argv += optind;

	if (argc == 1)
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

    if (!password && use_password)
    {
        password = prompt_password(keychainName);
		if (!password)
		{
			result = -1;
			goto loser;
		}
		zero_password = 1;
    }

	result = do_unlock(keychainName, password, use_password);
	if (result)
		goto loser;

loser:
	if (zero_password)
		memset(password, 0, strlen(password));

	return result;
}
