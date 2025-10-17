/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 20, 2025.
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
    @header SecItemInternal
    SecItemInternal defines SPI functions dealing with persistent refs
*/

#ifndef _SECURITY_SECITEMINTERNAL_H_
#define _SECURITY_SECITEMINTERNAL_H_

#include <CoreFoundation/CFData.h>
#include <sqlite3.h>

__BEGIN_DECLS

#define kSecServerKeychainChangedNotification   "com.apple.security.keychainchanged"
#define kSecServerCertificateTrustNotification  "com.apple.security.certificatetrust"
#define kSecServerSharedItemsChangedNotification "com.apple.security.shared-items-changed"

/* label when certificate data is joined with key data */
static const CFStringRef kSecAttrIdentityCertificateData = CFSTR("certdata");
static const CFStringRef kSecAttrIdentityCertificateTokenID = CFSTR("certtkid");

// Keys for dictionary of kSecvalueData of token-based items.
static const CFStringRef kSecTokenValueObjectIDKey = CFSTR("oid");
static const CFStringRef kSecTokenValueAccessControlKey = CFSTR("ac");
static const CFStringRef kSecTokenValueDataKey = CFSTR("data");

CFDataRef _SecItemCreatePersistentRef(CFTypeRef iclass, sqlite_int64 rowid, CFDictionaryRef attributes);
CFDataRef _SecItemCreateUUIDBasedPersistentRef(CFTypeRef iclass, CFDataRef uuidData, CFDictionaryRef attributes);

bool _SecItemParsePersistentRef(CFDataRef persistent_ref, CFStringRef *return_class,
    sqlite_int64 *return_rowid, CFDataRef *return_uuid, CFDictionaryRef *return_token_attrs);

OSStatus _SecRestoreKeychain(const char *path);

OSStatus SecOSStatusWith(bool (^perform)(CFErrorRef *error));

/* Structure representing copy-on-write dictionary.  Typical use is:
 int bar(CFDictionaryRef input);
 int foo(CFDictionaryRef input) {
     SecCFDictionaryCOW in = { input };
     if (condition) {
         CFDictionarySetValue(SecCFDictionaryCOWGetMutable(&in), key, value);
     }
     bar(in.dictionary);
     CFReleaseSafe(in.mutable_dictionary);
 }
 */
typedef struct {
    // Real dictionary, not owned by this structure, should be accessed directly for read-only access.
    CFDictionaryRef dictionary;

    // On-demand created (and possibly modified), owned writable copy of dictionary.
    CFMutableDictionaryRef mutable_dictionary;
} SecCFDictionaryCOW;

CFMutableDictionaryRef SecCFDictionaryCOWGetMutable(SecCFDictionaryCOW *cow_dictionary);

typedef enum {
    kSecItemAuthResultOK,
    kSecItemAuthResultError,
    kSecItemAuthResultNeedAuth
} SecItemAuthResult;

void SecItemAuthCopyParams(SecCFDictionaryCOW *auth_params, SecCFDictionaryCOW *query);

CFDictionaryRef SecTokenItemValueCopy(CFDataRef db_value, CFErrorRef *error);

CFArrayRef SecItemCopyParentCertificates_ios(CFDataRef normalizedIssuer, CFArrayRef accessGroups, CFErrorRef *error);

bool SecItemCertificateExists(CFDataRef normalizedIssuer, CFDataRef serialNumber, CFArrayRef accessGroups, CFErrorRef *error);

/*!
    @constant kSecAttrAppClipItem Boolean attribute indicating whether the origin of this item is an App Clip client
*/
static const CFStringRef kSecAttrAppClipItem = CFSTR("clip");

__END_DECLS

#endif /* !_SECURITY_SECITEMINTERNAL_H_ */
