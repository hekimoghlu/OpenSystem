/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 31, 2024.
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
/*!
 @header SecKeybagSupport.h - The thing that does the stuff with the gibli.
 */

#ifndef _SECURITYD_SECKEYBAGSUPPORT_H_
#define _SECURITYD_SECKEYBAGSUPPORT_H_

#include <CoreFoundation/CoreFoundation.h>
#include "utilities/SecAKSWrappers.h"
#include <libaks_acl_cf_keys.h>
#include <TargetConditionals.h>

#ifndef USE_KEYSTORE
// Use keystore (real or mock) on all platforms, except bridge
#define USE_KEYSTORE !TARGET_OS_BRIDGE
#endif

#if __has_include(<Kernel/IOKit/crypto/AppleKeyStoreDefs.h>)
#include <Kernel/IOKit/crypto/AppleKeyStoreDefs.h>
#endif

#if USE_KEYSTORE
#include <Security/SecAccessControlPriv.h>
#endif /* USE_KEYSTORE */

__BEGIN_DECLS


/* KEYBAG_NONE is private to security and have special meaning.
 They should not collide with AppleKeyStore constants, but are only referenced
 in here.
 */
#define KEYBAG_NONE (-1)   /* Set q_keybag to KEYBAG_NONE to obtain cleartext data. */
#define KEYBAG_DEVICE (g_keychain_keybag) /* actual keybag used to encrypt items */
extern keybag_handle_t g_keychain_keybag;

bool use_hwaes(void);

bool ks_crypt(CFTypeRef operation, keybag_handle_t keybag, struct backup_keypair* bkp,
              keyclass_t keyclass, uint32_t textLength, const uint8_t *source, keyclass_t *actual_class,
              CFMutableDataRef dest, bool useNewBackupBehavior, CFErrorRef *error);
bool ks_crypt_diversify(CFTypeRef operation, keybag_handle_t keybag,
              keyclass_t keyclass, uint32_t textLength, const uint8_t *source, keyclass_t *actual_class,
              CFMutableDataRef dest, const void *personaId, size_t personaIdLength, CFErrorRef *error);
bool ks_separate_data_and_key(CFDictionaryRef blob_dict, CFDataRef *ed_data, CFDataRef *key_data);

#if USE_KEYSTORE
bool ks_encrypt_acl(keybag_handle_t keybag, keyclass_t keyclass, uint32_t textLength, const uint8_t *source,
                  CFMutableDataRef dest, CFDataRef auth_data, CFDataRef acm_context,
                  SecAccessControlRef access_control, CFErrorRef *error);
bool ks_decrypt_acl(aks_ref_key_t ref_key, CFDataRef encrypted_data, CFMutableDataRef dest,
                    CFDataRef acm_context, CFDataRef caller_access_groups,
                    SecAccessControlRef access_control, CFErrorRef *error);
bool ks_delete_acl(aks_ref_key_t ref_key, CFDataRef encrypted_data,
                   CFDataRef acm_context, CFDataRef caller_access_groups,
                   SecAccessControlRef access_control, CFErrorRef *error);
const void* ks_ref_key_get_external_data(keybag_handle_t keybag, CFDataRef key_data,
                                         aks_ref_key_t *ref_key, size_t *external_data_len, CFErrorRef *error);

bool ks_access_control_needed_error(CFErrorRef *error, CFDataRef access_control_data, CFTypeRef operation);
bool create_cferror_from_aks(int aks_return, CFTypeRef operation, keybag_handle_t keybag, keyclass_t keyclass, CFDataRef access_control_data, CFDataRef acm_context_data, CFErrorRef *error);
#endif
bool ks_open_keybag(CFDataRef keybag, CFDataRef password, keybag_handle_t *handle, CFErrorRef *error);
bool ks_close_keybag(keybag_handle_t keybag, CFErrorRef *error);
__END_DECLS

#endif /* _SECURITYD_SECKEYBAGSUPPORT_H_ */
