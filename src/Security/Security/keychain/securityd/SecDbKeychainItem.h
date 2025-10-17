/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 30, 2023.
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
 @header SecDbKeychainItem.h - The thing that does the stuff with the gibli.
 */

#ifndef _SECURITYD_SECKEYCHAINITEM_H_
#define _SECURITYD_SECKEYCHAINITEM_H_

#include "keychain/securityd/SecKeybagSupport.h"
#include "keychain/securityd/SecDbItem.h"
#include "keychain/securityd/SecDbQuery.h"

CF_ASSUME_NONNULL_BEGIN

__BEGIN_DECLS

bool ks_encrypt_data(keybag_handle_t keybag, SecAccessControlRef _Nullable access_control, CFDataRef _Nullable acm_context,
                     CFDictionaryRef secretData, CFDictionaryRef attributes, CFDictionaryRef authenticated_attributes, CFDataRef _Nullable *_Nonnull pBlob, bool useDefaultIV,
                     bool useNewBackupBehavior, CFErrorRef _Nullable *_Nullable error);
bool ks_encrypt_data_legacy(keybag_handle_t keybag, SecAccessControlRef _Nullable access_control, CFDataRef _Nullable acm_context,
                            CFDictionaryRef attributes, CFDictionaryRef authenticated_attributes, CFDataRef _Nullable *_Nonnull pBlob, bool useDefaultIV,
                            bool useNewBackupBehavior, CFErrorRef _Nullable *_Nullable error); // used for backup
bool ks_decrypt_data(keybag_handle_t keybag, struct backup_keypair *_Nullable bkp, CFTypeRef cryptoOp, SecAccessControlRef _Nullable *_Nullable paccess_control, CFDataRef acm_context,
                     CFDataRef blob, const SecDbClass *db_class, CFArrayRef caller_access_groups,
                     CFMutableDictionaryRef _Nullable *_Nullable attributes_p, uint32_t *_Nullable version_p, bool decryptSecretData, keyclass_t*_Nullable outKeyclass, CFErrorRef _Nullable *_Nullable error);
bool s3dl_item_from_data(CFDataRef edata, Query *q, CFArrayRef accessGroups,
                         CFMutableDictionaryRef _Nonnull *_Nonnull item, SecAccessControlRef _Nullable *_Nullable access_control, keyclass_t *_Nullable keyclass, CFErrorRef _Nullable *_Nullable error);
SecDbItemRef _Nullable SecDbItemCreateWithBackupDictionary(const SecDbClass *dbclass, CFDictionaryRef dict, keybag_handle_t src_keybag, struct backup_keypair *_Nullable src_bkp, keybag_handle_t dst_keybag, CFErrorRef _Nullable *_Nullable error);
bool SecDbItemExtractRowIdFromBackupDictionary(SecDbItemRef item, CFDictionaryRef dict, CFErrorRef _Nullable *_Nullable error);
bool SecDbItemExtractUUIDPersistentRefFromBackupDictionary(SecDbItemRef item, CFDictionaryRef dict, CFErrorRef *error);
bool SecDbItemInferSyncable(SecDbItemRef item, CFErrorRef _Nullable *_Nullable error);

CFTypeRef _Nullable SecDbKeychainItemCopyPrimaryKey(SecDbItemRef item, const SecDbAttr *attr, CFErrorRef _Nullable *_Nullable error);
CFTypeRef _Nullable SecDbKeychainItemCopySHA256PrimaryKey(SecDbItemRef item, CFErrorRef _Nullable *_Nullable error);
CFTypeRef _Nullable SecDbKeychainItemCopySHA1(SecDbItemRef item, const SecDbAttr *attr, CFErrorRef _Nullable *_Nullable error);
CFTypeRef _Nullable SecDbKeychainItemCopyCurrentDate(SecDbItemRef item, const SecDbAttr *attr, CFErrorRef _Nullable *_Nullable error);
CFTypeRef _Nullable SecDbKeychainItemCopyEncryptedData(SecDbItemRef item, const SecDbAttr *attr, CFErrorRef _Nullable *_Nullable error);

SecAccessControlRef _Nullable SecDbItemCopyAccessControl(SecDbItemRef item, CFErrorRef _Nullable *_Nullable error);
bool SecDbItemSetAccessControl(SecDbItemRef item, SecAccessControlRef access_control, CFErrorRef _Nullable *_Nullable error);

void SecDbResetMetadataKeys(void);

__END_DECLS

CF_ASSUME_NONNULL_END

#endif /* _SECURITYD_SECKEYCHAINITEM_H_ */
