/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 27, 2023.
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
#include "keychain_set_settings.h"
#include "keychain_utilities.h"
#include "readline_cssm.h"
#include "security_tool.h"

#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <Security/SecKeychain.h>
#include <Security/SecKeychainPriv.h>

#define PW_BUF_SIZE 512				/* size of buffer to alloc for password */


static int
do_keychain_set_settings(const char *keychainName, SecKeychainSettings newKeychainSettings)
{
	SecKeychainRef keychain = NULL;
	OSStatus result;

	if (keychainName)
	{
		keychain = keychain_open(keychainName);
		if (!keychain)
		{
			result = 1;
			goto cleanup;
		}
	}
	result = SecKeychainSetSettings(keychain, &newKeychainSettings);
	if (result)
	{
		sec_error("SecKeychainSetSettings %s: %s", keychainName ? keychainName : "<NULL>", sec_errstr(result));
	}

cleanup:
	if (keychain)
		CFRelease(keychain);

	return result;
}


static int
do_keychain_set_password(const char *keychainName, const char* oldPassword, const char* newPassword)
{
	SecKeychainRef keychain = NULL;
	OSStatus result = 1;
	UInt32 oldLen = (oldPassword) ? (UInt32) strlen(oldPassword) : 0;
	UInt32 newLen = (newPassword) ? (UInt32) strlen(newPassword) : 0;
	char *oldPass = (oldPassword) ? (char*)oldPassword : NULL;
	char *newPass = (newPassword) ? (char*)newPassword : NULL;
	char *oldBuf = NULL;
	char *newBuf = NULL;

	if (keychainName)
	{
		keychain = keychain_open(keychainName);
		if (!keychain)
		{
			result = 1;
			goto cleanup;
		}
	}

	if (!oldPass) {
		/* prompt for old password */
		char *pBuf = getpass("Old Password: ");
		if (pBuf) {
			oldBuf = (char*) calloc(PW_BUF_SIZE, 1);
			oldLen = (UInt32) strlen(pBuf);
			memcpy(oldBuf, pBuf, oldLen);
			bzero(pBuf, oldLen);
			oldPass = oldBuf;
		}
	}

	if (!newPass) {
		/* prompt for new password */
		char *pBuf = getpass("New Password: ");
		if (pBuf) {
			newBuf = (char*) calloc(PW_BUF_SIZE, 1);
			newLen = (UInt32) strlen(pBuf);
			memcpy(newBuf, pBuf, newLen);
			bzero(pBuf, newLen);
		}
		/* confirm new password */
		pBuf = getpass("Retype New Password: ");
		if (pBuf) {
			UInt32 confirmLen = (UInt32) strlen(pBuf);
			if (confirmLen == newLen && newBuf &&
				!memcmp(pBuf, newBuf, newLen)) {
				newPass = newBuf;
			}
			bzero(pBuf, confirmLen);
		}
	}

	if (!oldPass || !newPass) {
		sec_error("try again");
		goto cleanup;
	}

	/* change the password, if daemon agrees everything looks good */
	result = SecKeychainChangePassword(keychain, oldLen, oldPass, newLen, newPass);
	if (result)
	{
		sec_error("error changing password for \"%s\": %s",
			keychainName ? keychainName : "<NULL>", sec_errstr(result));
	}

cleanup:
	/* if we allocated password buffers, zero and free them */
	if (oldBuf) {
		bzero(oldBuf, PW_BUF_SIZE);
		free(oldBuf);
	}
	if (newBuf) {
		bzero(newBuf, PW_BUF_SIZE);
		free(newBuf);
	}
	if (keychain) {
		CFRelease(keychain);
	}
	return result;
}


int
keychain_set_settings(int argc, char * const *argv)
{
	char *keychainName = NULL;
	int ch, result = 0;
    SecKeychainSettings newKeychainSettings =
		{ SEC_KEYCHAIN_SETTINGS_VERS1, FALSE, FALSE, INT_MAX };

    while ((ch = getopt(argc, argv, "hlt:u")) != -1)
	{
		switch  (ch)
		{
        case 'l':
            newKeychainSettings.lockOnSleep = TRUE;
			break;
		case 't':
            newKeychainSettings.lockInterval = atoi(optarg);
			break;
		case 'u':
            newKeychainSettings.useLockInterval = TRUE;
			break;
		case '?':
		default:
			result = 2; /* @@@ Return 2 triggers usage message. */
			goto cleanup;
		}
	}

	if (newKeychainSettings.lockInterval != INT_MAX) {
		// -t was specified, which implies -u
		newKeychainSettings.useLockInterval = TRUE;
	} else {
		// -t was unspecified, so revert to no timeout
		newKeychainSettings.useLockInterval = FALSE;
	}

	argc -= optind;
	argv += optind;

	if (argc == 1)
	{
		keychainName = argv[0];
		if (*keychainName == '\0')
		{
			result = 2;
			goto cleanup;
		}
	}
	else if (argc != 0)
	{
		result = 2;
		goto cleanup;
	}

	result = do_keychain_set_settings(keychainName, newKeychainSettings);

cleanup:

	return result;
}

int
keychain_set_password(int argc, char * const *argv)
{
	char *keychainName = NULL;
	char *oldPassword = NULL;
	char *newPassword = NULL;
	int ch, result = 0;

    while ((ch = getopt(argc, argv, "ho:p:")) != -1)
	{
		switch  (ch)
		{
        case 'o':
            oldPassword = optarg;
			break;
        case 'p':
            newPassword = optarg;
			break;
		case '?':
		default:
			result = 2; /* @@@ Return 2 triggers usage message. */
			goto cleanup;
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
			goto cleanup;
		}
	}
	else if (argc != 0)
	{
		result = 2;
		goto cleanup;
	}

	result = do_keychain_set_password(keychainName, oldPassword, newPassword);

cleanup:

	return result;
}

