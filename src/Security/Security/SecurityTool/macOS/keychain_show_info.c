/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 9, 2022.
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
#include "keychain_show_info.h"
#include "keychain_utilities.h"
#include "readline_cssm.h"
#include "security_tool.h"

#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <Security/SecKeychain.h>

static int
do_keychain_show_info(const char *keychainName)
{
	SecKeychainRef keychain = NULL;
    SecKeychainSettings keychainSettings = { SEC_KEYCHAIN_SETTINGS_VERS1 };
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

	result = SecKeychainCopySettings(keychain, &keychainSettings);
	if (result)
	{
		sec_error("SecKeychainCopySettings %s: %s", keychainName ? keychainName : "<NULL>", sec_errstr(result));
		goto loser;
	}

    fprintf(stderr,"Keychain \"%s\"%s%s",
		keychainName ? keychainName : "<NULL>",
		keychainSettings.lockOnSleep ? " lock-on-sleep" : "",
		keychainSettings.useLockInterval ? " use-lock-interval" : "");
	if (keychainSettings.lockInterval == INT_MAX)
		fprintf(stderr," no-timeout\n");
	else
		fprintf(stderr," timeout=%ds\n", (int)keychainSettings.lockInterval);

loser:
	if (keychain)
		CFRelease(keychain);
	return result;
}

int
keychain_show_info(int argc, char * const *argv)
{
	char *keychainName = NULL;
	int  result = 0;

	if (argc == 2)
	{
		keychainName = argv[1];
		if (*keychainName == '\0')
		{
			result = 2;
			goto loser;
		}
	}
	else if (argc != 1)
		return SHOW_USAGE_MESSAGE;

	result = do_keychain_show_info(keychainName);

loser:
	return result;
}
