/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 8, 2023.
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
#import <Foundation/Foundation.h>
#import <Security/Security.h>
#import <Security/SecItemPriv.h>
#include <sys/stat.h>
#include <stdio.h>

#include "xmalloc.h"
#include "sshkey.h"
#include "ssherr.h"
#include "authfile.h"
#include "openbsd-compat/openbsd-compat.h"
#include "log.h"

char *keychain_read_passphrase(const char *filename)
{
	OSStatus	ret = errSecSuccess;
	NSString	*accountString = [NSString stringWithUTF8String: filename];
	NSData		*passphraseData = NULL;

	if (accountString == nil) {
		debug2("Cannot retrieve identity passphrase from the keychain since the path is not UTF8.");
		return NULL;
	}

	NSDictionary	*searchQuery = @{
			       (id)kSecClass: (id)kSecClassGenericPassword,
			       (id)kSecAttrAccount: accountString,
			       (id)kSecAttrLabel: [NSString stringWithFormat: @"SSH: %@", accountString],
			       (id)kSecAttrService: @"OpenSSH",
			       (id)kSecUseDataProtectionKeychain: @YES,
			       (id)kSecAttrAccessGroup: @"com.apple.ssh.passphrases",
			       (id)kSecReturnData: @YES,
			       (id)kSecUseAuthenticationUI: (id)kSecUseAuthenticationUIFail};
	debug3("Search for item with query: %s", [[searchQuery description] UTF8String]);
	ret = SecItemCopyMatching((CFDictionaryRef)searchQuery, (CFTypeRef *)&passphraseData);
	if (ret == errSecItemNotFound) {
		debug2("Passphrase not found in the keychain.");
		return NULL;
	} else if (ret != errSecSuccess) {
		NSString *errorString = (NSString *)SecCopyErrorMessageString(ret, NULL);
		debug2("Unexpected keychain error while searching for an item: %s", [errorString UTF8String]);
		[errorString release];
		[passphraseData release];
		return NULL;
	}

	if (![passphraseData isKindOfClass: [NSData class]]) {
		debug2("Malformed result returned from the keychain");
		[passphraseData release];
		return NULL;
	}

	char *passphrase = xcalloc([passphraseData length] + 1, sizeof(char));
	[passphraseData getBytes: passphrase length: [passphraseData length]];
	[passphraseData release];

	// Try to load the key first and only return the passphrase if we know it's the right one
	struct sshkey *private = NULL;
	int r = sshkey_load_private_type(KEY_UNSPEC, filename, passphrase, &private, NULL);
	if (r != SSH_ERR_SUCCESS) {
		debug2("Could not unlock key with the passphrase retrieved from the keychain.");
		freezero(passphrase, strlen(passphrase));
		return NULL;
	}
	sshkey_free(private);

	return passphrase;
}

void store_in_keychain(const char *filename, const char *passphrase)
{
	OSStatus	ret = errSecSuccess;
	BOOL		updateExistingItem = NO;
	NSString	*accountString = [NSString stringWithUTF8String: filename];

	if (accountString == nil) {
		debug2("Cannot store identity passphrase into the keychain since the path is not UTF8.");
		return;
	}

	NSDictionary	*defaultAttributes = @{
				(id)kSecClass: (id)kSecClassGenericPassword,
				(id)kSecAttrAccount: accountString,
				(id)kSecAttrLabel: [NSString stringWithFormat: @"SSH: %@", accountString],
				(id)kSecAttrService: @"OpenSSH",
				(id)kSecUseDataProtectionKeychain: @YES,
				(id)kSecAttrAccessGroup: @"com.apple.ssh.passphrases",
				(id)kSecUseAuthenticationUI: (id)kSecUseAuthenticationUIFail};

	CFTypeRef searchResults = NULL;
	NSMutableDictionary *searchQuery = [@{(id)kSecReturnRef: @YES} mutableCopy];
	[searchQuery addEntriesFromDictionary: defaultAttributes];

	debug3("Search for existing item with query: %s", [[searchQuery description] UTF8String]);
	ret = SecItemCopyMatching((CFDictionaryRef)searchQuery, &searchResults);
	[searchQuery release];
	if (ret == errSecSuccess) {
		debug3("Item already exists in the keychain, updating.");
		updateExistingItem = YES;

	} else if (ret == errSecItemNotFound) {
		debug3("Item does not exist in the keychain, adding.");
	} else {
		NSString *errorString = (NSString *)SecCopyErrorMessageString(ret, NULL);
		debug3("Unexpected keychain error while searching for an item: %s", [errorString UTF8String]);
		[errorString release];
	}

	if (updateExistingItem) {
		NSDictionary *updateQuery = defaultAttributes;
		NSDictionary *changes = @{(id)kSecValueData: [NSData dataWithBytesNoCopy: (void *)passphrase length: strlen(passphrase) freeWhenDone: NO]};

		ret = SecItemUpdate((CFDictionaryRef)updateQuery, (CFDictionaryRef)changes);
		if (ret != errSecSuccess) {
			NSString *errorString = (NSString *)SecCopyErrorMessageString(ret, NULL);
			debug3("Unexpected keychain error while updating the item: %s", [errorString UTF8String]);
			[errorString release];
		}
	} else {
		NSMutableDictionary *addQuery = [@{(id)kSecValueData: [NSData dataWithBytesNoCopy: (void *)passphrase length: strlen(passphrase) freeWhenDone: NO]} mutableCopy];

		[addQuery addEntriesFromDictionary: defaultAttributes];
		ret = SecItemAdd((CFDictionaryRef)addQuery, NULL);
		[addQuery release];
		if (ret != errSecSuccess) {
			NSString *errorString = (NSString *)SecCopyErrorMessageString(ret, NULL);
			debug3("Unexpected keychain error while inserting the item: %s", [errorString UTF8String]);
			[errorString release];
		}
	}
}

/*
 * Remove the passphrase for a given identity from the keychain.
 */
void
remove_from_keychain(const char *filename)
{
	OSStatus	ret = errSecSuccess;
	NSString	*accountString = [NSString stringWithUTF8String: filename];

	if (accountString == nil) {
		debug2("Cannot delete identity passphrase from the keychain since the path is not UTF8.");
		return;
	}

	NSDictionary	*searchQuery = @{
			       (id)kSecClass: (id)kSecClassGenericPassword,
			       (id)kSecAttrAccount: accountString,
			       (id)kSecAttrService: @"OpenSSH",
			       (id)kSecUseDataProtectionKeychain: @YES,
			       (id)kSecAttrAccessGroup: @"com.apple.ssh.passphrases",
			       (id)kSecUseAuthenticationUI: (id)kSecUseAuthenticationUIFail};

	ret = SecItemDelete((CFDictionaryRef)searchQuery);
	if (ret == errSecSuccess) {
		NSString *errorString = (NSString *)SecCopyErrorMessageString(ret, NULL);
		debug3("Unexpected keychain error while deleting the item: %s", [errorString UTF8String]);
		[errorString release];
	}
}


int
load_identities_from_keychain(int (^add_identity)(const char *identity))
{
	int 		ret = 0;
	OSStatus	err = errSecSuccess;

	NSArray		*searchResults = nil;
	NSDictionary	*searchQuery = @{
				(id)kSecClass: (id)kSecClassGenericPassword,
				(id)kSecAttrService: @"OpenSSH",
				(id)kSecUseDataProtectionKeychain: @YES,
				(id)kSecAttrAccessGroup: @"com.apple.ssh.passphrases",
				(id)kSecReturnAttributes: @YES,
				(id)kSecMatchLimit: (id)kSecMatchLimitAll,
				(id)kSecUseAuthenticationUI: (id)kSecUseAuthenticationUIFail};

	err = SecItemCopyMatching((CFDictionaryRef)searchQuery, (CFTypeRef *)&searchResults);
	if (err == errSecItemNotFound) {
		fprintf(stderr, "No identity found in the keychain.\n");
		[searchResults release];
		return 0;
	} else if (err != errSecSuccess || ![searchResults isKindOfClass: [NSArray class]]) {
		return 1;
	}

	for (NSDictionary *itemAttributes in searchResults) {
		NSString	*accountString = itemAttributes[(id)kSecAttrAccount];
		struct stat	st;

		if (stat([accountString UTF8String], &st) < 0)
			continue;
		if (add_identity([accountString UTF8String]))
			ret = 1;
	}
	[searchResults release];

	return ret;
}

void
warn_keychain_option()
{
	static int warned = 0;
	if (warned++)
		return;
	fprintf(stderr,
"WARNING: The -K and -A flags are deprecated and have been replaced\n"
"         by the --apple-use-keychain and --apple-load-keychain\n"
"         flags, respectively.  To suppress this warning, set the\n"
"         environment variable APPLE_SSH_ADD_BEHAVIOR as described in\n"
"         the ssh-add(1) manual page.\n");
}
