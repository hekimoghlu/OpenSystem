/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 14, 2022.
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
#ifndef _KEYCHAIN_FINDINTERNETPASSWORD_H_
#define _KEYCHAIN_FINDINTERNETPASSWORD_H_  1

#include <Security/SecBase.h>
#include <Security/SecKeychain.h>

#ifdef __cplusplus
extern "C" {
#endif

extern SecKeychainItemRef find_first_generic_password(
							CFTypeRef keychainOrArray,
							FourCharCode itemCreator,
							FourCharCode itemType,
							const char *kind,
							const char *value,
							const char *comment,
							const char *label,
							const char *serviceName,
							const char *accountName);

extern SecKeychainItemRef find_first_internet_password(
							CFTypeRef keychainOrArray,
							FourCharCode itemCreator,
							FourCharCode itemType,
							const char *kind,
							const char *comment,
							const char *label,
							const char *serverName,
							const char *securityDomain,
							const char *accountName,
							const char *path,
							UInt16 port,
							SecProtocolType protocol,
							SecAuthenticationType authenticationType);

extern SecKeychainItemRef find_unique_certificate(
							CFTypeRef keychainOrArray,
							const char *name,
							const char *hash);

extern int keychain_find_internet_password(int argc, char * const *argv);

extern int keychain_delete_internet_password(int argc, char * const *argv);

extern int keychain_find_generic_password(int argc, char * const *argv);

extern int keychain_delete_generic_password(int argc, char * const *argv);

extern int keychain_find_key(int argc, char* const *argv);

extern int keychain_set_generic_password_partition_list(int argc, char * const *argv);

extern int keychain_set_internet_password_partition_list(int argc, char * const *argv);

extern int keychain_set_key_partition_list(int argc, char * const *argv);

extern int keychain_find_certificate(int argc, char * const *argv);

extern int keychain_dump(int argc, char * const *argv);

#ifdef __cplusplus
}
#endif

#endif /* _KEYCHAIN_FINDINTERNETPASSWORD_H_ */
