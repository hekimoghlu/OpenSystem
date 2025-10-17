/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 14, 2024.
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
#ifndef _KEYCHAIN_UTILITIES_H_
#define _KEYCHAIN_UTILITIES_H_ 1

#include <Security/SecKeychain.h>

#include <stdio.h>
#include <CoreFoundation/CFData.h>
#include <CoreFoundation/CFDictionary.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Open a given named keychain. */
extern SecKeychainRef keychain_open(const char *name);

/* Return either NULL if argc is 0, or a SecKeychainRef if argc is 1 or a CFArrayRef of SecKeychainRefs if argc is greater than 1. */
extern CFTypeRef keychain_create_array(int argc, char * const *argv);

extern int parse_fourcharcode(const char *name, UInt32 *code);

extern int print_keychain_name(FILE *stream, SecKeychainRef keychain);

extern int print_keychain_item_attributes(FILE *stream, SecKeychainItemRef item, Boolean show_data, Boolean show_raw_data, Boolean show_acl, Boolean interactive);

extern void print_cfstring(FILE *stream, CFStringRef string);

extern void print_buffer(FILE *stream, size_t length, const void *data);

extern void print_buffer_pem(FILE *stream, const char *headerString, size_t length, const void *data);

extern void print_uint32(FILE* stream, uint32 n);

extern unsigned char hexToValue(char c, char *error);
extern unsigned char hexValue(char c);

extern bool convertHex(const char *hexString, uint8_t **outData, size_t *outLength);

extern void fromHex(const char *hexDigits, CSSM_DATA *data);

extern CFDataRef cfFromHex(CFStringRef hex);

extern CFStringRef cfToHex(CFDataRef bin);

extern CFDictionaryRef makeCFDictionaryFromData(CFDataRef data);

extern void GetCStringFromCFString(CFStringRef cfstring, char** cstr, size_t* len);

extern void print_partition_id_list(FILE *stream, CFStringRef description);

extern void safe_CFRelease(void CF_CONSUMED *cfTypeRefPtr);

extern void check_obsolete_keychain(const char *kcName);

extern char* prompt_password(const char* keychainName);

#ifdef __cplusplus
}
#endif

#endif /*  _KEYCHAIN_UTILITIES_H_ */
