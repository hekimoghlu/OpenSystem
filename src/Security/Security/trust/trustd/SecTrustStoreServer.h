/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 20, 2025.
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
    @header SecTrustStoreServer
    CertificateSource API to a system root certificate store
*/

#ifndef _SECURITY_SECTRUSTSTORESERVER_H_
#define _SECURITY_SECTRUSTSTORESERVER_H_

#include "Security/SecTrustStore.h"
#include <CoreFoundation/CFArray.h>
#include <CoreFoundation/CFError.h>

__BEGIN_DECLS

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wnullability-completeness"

SecTrustStoreRef SecTrustStoreForDomainName(CFStringRef domainName, CFErrorRef *error);

bool _SecTrustStoreSetTrustSettings(SecTrustStoreRef ts, SecCertificateRef certificate, CFTypeRef trustSettingsDictOrArray, CFErrorRef *error);

bool _SecTrustStoreRemoveCertificate(SecTrustStoreRef ts, SecCertificateRef cert, CFErrorRef *error);

bool _SecTrustStoreRemoveAll(SecTrustStoreRef ts, CFErrorRef *error);

CFArrayRef SecTrustStoreCopyParents(SecTrustStoreRef ts, SecCertificateRef certificate, CFErrorRef *error);

bool _SecTrustStoreContainsCertificate(SecTrustStoreRef source, SecCertificateRef cert, bool *contains, CFErrorRef *error);

bool _SecTrustStoreCopyUsageConstraints(SecTrustStoreRef ts, SecCertificateRef cert, CFArrayRef *usageConstraints, CFErrorRef *error);

bool _SecTrustStoreCopyAll(SecTrustStoreRef ts, CFStringRef policyId, CFArrayRef *trustStoreContents, CFErrorRef *error);

bool _SecTrustStoreMigrateUserStore(CFErrorRef *error);

void _SecTrustStoreMigrateConfigurations(void);

void _SecTrustStoreMigrateTrustSettings(void);

bool _SecTrustStoreMigrateTrustSettingsPropertyList(CFErrorRef *error);

#pragma clang diagnostic pop

void SecTrustStoreMigratePropertyListBlock(uid_t uid, CFPropertyListRef _Nullable plist, CFDictionaryRef _Nullable certificates, void (^ _Nonnull completed)(bool result, CFErrorRef _Nullable error));

bool SecTrustStoreMigratePropertyList(uid_t uid, CFPropertyListRef _Nullable plist, CFDictionaryRef _Nullable certificates, CFErrorRef _Nonnull * _Nullable error);

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wnullability-completeness"

bool _SecTrustStoreSetCTExceptions(CFStringRef appID, CFDictionaryRef exceptions, CFErrorRef *error);
CF_RETURNS_RETAINED CFDictionaryRef _SecTrustStoreCopyCTExceptions(CFStringRef appID, CFErrorRef *error);

bool _SecTrustStoreSetCARevocationAdditions(CFStringRef appID, CFDictionaryRef additions, CFErrorRef *error);
CF_RETURNS_RETAINED CFDictionaryRef _SecTrustStoreCopyCARevocationAdditions(CFStringRef appID, CFErrorRef *error);

bool _SecTrustStoreSetTransparentConnectionPins(CFStringRef appID, CFArrayRef pins, CFErrorRef *error);
CF_RETURNS_RETAINED CFArrayRef _SecTrustStoreCopyTransparentConnectionPins(CFStringRef appID, CFErrorRef *error);

#pragma clang diagnostic pop

__END_DECLS

#endif /* !_SECURITY_SECTRUSTSTORESERVER_H_ */
