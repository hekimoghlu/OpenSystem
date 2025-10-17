/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 25, 2024.
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
    @header SecTrustStore
    CertificateSource API to a system root certificate store
*/

#ifndef _SECURITY_SECTRUSTSTORE_H_
#define _SECURITY_SECTRUSTSTORE_H_

#include <Security/SecCertificate.h>
#include <CoreFoundation/CoreFoundation.h>

__BEGIN_DECLS

typedef struct __SecTrustStore *SecTrustStoreRef;

enum {
	kSecTrustStoreDomainSystem = 1,
	kSecTrustStoreDomainUser = 2,
	kSecTrustStoreDomainAdmin = 3,
};
typedef uint32_t SecTrustStoreDomain;

typedef int32_t SecTrustSettingsVersionNumber;

typedef int32_t SecTrustSettingsAssetVersionNumber;

SecTrustStoreRef SecTrustStoreForDomain(SecTrustStoreDomain domain);

CFStringRef SecTrustStoreDomainName(SecTrustStoreRef ts);

Boolean SecTrustStoreContains(SecTrustStoreRef source, SecCertificateRef certificate);

/* Only allowed for writable trust stores. */
OSStatus SecTrustStoreSetTrustSettings(SecTrustStoreRef ts,
	SecCertificateRef certificate,
    CFTypeRef trustSettingsDictOrArray);

OSStatus SecTrustStoreRemoveCertificate(SecTrustStoreRef ts,
	SecCertificateRef certificate);

OSStatus SecTrustStoreRemoveAll(SecTrustStoreRef ts);

OSStatus SecTrustStoreGetSettingsVersionNumber(SecTrustSettingsVersionNumber* p_settings_version_number);

OSStatus SecTrustStoreGetSettingsAssetVersionNumber(SecTrustSettingsAssetVersionNumber* p_settings_asset_version_number);

OSStatus SecTrustStoreCopyAll(SecTrustStoreRef ts, CFArrayRef *CF_RETURNS_RETAINED trustStoreContents);

/* Note that usageConstraints may be NULL on success. */
OSStatus SecTrustStoreCopyUsageConstraints(SecTrustStoreRef ts,
	SecCertificateRef certificate,
	CFArrayRef *CF_RETURNS_RETAINED usageConstraints);

CFArrayRef SecTrustStoreCopyAnchors(SecTrustStoreRef ts, CFStringRef policyId);

__END_DECLS

#endif /* !_SECURITY_SECTRUSTSTORE_H_ */
