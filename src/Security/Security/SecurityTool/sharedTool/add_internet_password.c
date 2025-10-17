/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 10, 2023.
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
#include <string.h>
#include <getopt.h>
#include <stdlib.h>

#include <Security/SecItem.h>
#include <CoreFoundation/CFString.h>
#include <CoreFoundation/CFNumber.h>

#include "SecurityTool/sharedTool/tool_errors.h"

#include "SecurityCommands.h"

#include "tool_auth_helpers.h"

typedef uint32_t SecProtocolType;
typedef uint32_t SecAuthenticationType;

static int
do_addinternetpassword(const char *keychainName, const char *serverName,
                       const char *securityDomain, const char *accountName, const char *path,
                       UInt16 port, SecProtocolType protocol,
                       SecAuthenticationType authenticationType, const void *passwordData)
{
	OSStatus result;
    CFDictionaryRef attributes = NULL;
    const void *keys[9], *values[9];
    CFIndex ix = 0;
    
    keys[ix] = kSecClass;
    values[ix++] = kSecClassInternetPassword;
    
	if (serverName) {
		keys[ix] = kSecAttrServer;
		values[ix++] = CFStringCreateWithCStringNoCopy(NULL, serverName,
                                                       kCFStringEncodingUTF8, kCFAllocatorNull);
	}
	if (securityDomain) {
		keys[ix] = kSecAttrSecurityDomain;
		values[ix++] = CFStringCreateWithCStringNoCopy(NULL, securityDomain,
                                                       kCFStringEncodingUTF8, kCFAllocatorNull);
	}
	if (accountName) {
		keys[ix] = kSecAttrAccount;
		values[ix++] = CFStringCreateWithCStringNoCopy(NULL, accountName,
                                                       kCFStringEncodingUTF8, kCFAllocatorNull);
	}
	if (path) {
		keys[ix] = kSecAttrPath;
		values[ix++] = CFStringCreateWithCStringNoCopy(NULL, path,
                                                       kCFStringEncodingUTF8, kCFAllocatorNull);
	}
    keys[ix] = kSecAttrPort;
    values[ix++] = CFNumberCreate(NULL, kCFNumberSInt16Type, &port);
    /* Protocol is a 4 char code, perhaps we should use a string rep instead. */
    keys[ix] = kSecAttrProtocol;
    values[ix++] = CFNumberCreate(NULL, kCFNumberSInt32Type, &protocol);
    keys[ix] = kSecAttrAuthenticationType;
    values[ix++] = CFNumberCreate(NULL, kCFNumberSInt32Type, &authenticationType);
    
    if (passwordData)
    {
        keys[ix] = kSecValueData;
        values[ix++] = CFDataCreateWithBytesNoCopy(NULL, passwordData,
                                                   strlen(passwordData), kCFAllocatorNull);
    }
    
    attributes = CFDictionaryCreate(NULL, keys, values, ix,
                                    &kCFTypeDictionaryKeyCallBacks, &kCFTypeDictionaryValueCallBacks);
    
	/* Release the values we just added to the dictionary. */
	/* @@@ For performance reasons we could choose to make the dictionaries
     CFRetain callback a no op and let the dictionary deal with the
     releasing. */
    for (; ix-- > 0;) {
        CFRelease(values[ix]);
    }
    
	result = SecItemAdd(attributes, NULL);
    
    if (attributes) {
        CFRelease(attributes);
    }

	if (result)
	{
		sec_perror("SecItemAdd", result);
	}
    
	return result;
}


int keychain_add_internet_password(int argc, char * const *argv)
{
	char *serverName = NULL, *securityDomain = NULL, *accountName = NULL, *path = NULL, *passwordData  = NULL;
    UInt16 port = 0;
    SecProtocolType protocol = 0;
    SecAuthenticationType authenticationType = 0;
	int ch, result = 0;
	const char *keychainName = NULL;
    bool authSucceeded = false;
    /*
     -s Use servername\n"
     "    -e Use securitydomain\n"
     "    -a Use accountname\n"
     "    -p Use path\n"
     "    -o Use port \n"
     "    -r Use protocol \n"
     "    -c Use SecAuthenticationType  \n"
     "    -w Use passwordData  \n"
     */
	while ((ch = getopt(argc, argv, "s:d:a:p:P:r:t:w:hyY:")) != -1)
	{
		switch  (ch)
		{
            case 's':
                serverName = optarg;
                break;
            case 'd':
                securityDomain = optarg;
                break;
            case 'a':
                accountName = optarg;
                break;
            case 'p':
                path = optarg;
                break;
            case 'P':
                port = atoi(optarg);
                break;
            case 'r':
                memcpy(&protocol,optarg,4);
                //protocol = (SecProtocolType)optarg;
                break;
            case 't':
                memcpy(&authenticationType,optarg,4);
                break;
            case 'w':
                passwordData = optarg;
                break;
            case 'y':
                if (!promptForAndCheckPassphrase()) {
                    return 1;
                }
                authSucceeded = true;
                break;
            case 'Y':
                if (!checkPassphrase(optarg, 0) ) {
                    return 1;
                }
                authSucceeded = true;
                break;
            case '?':
            default:
                return SHOW_USAGE_MESSAGE;
		}
	}
    
    if (!authSucceeded && authRequired()) {
        return 1;
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
    
	result = do_addinternetpassword(keychainName, serverName, securityDomain,
                                    accountName, path, port, protocol,authenticationType, passwordData);
    
loser:
	return result;
}
