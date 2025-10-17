/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 17, 2024.
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
/**** This file is included from tls-darwin.c ****/

extern char **environ;

#ifndef _SECURITY_VERSION_GREATER_THAN_57610_
typedef CF_OPTIONS(uint32_t, SecKeyUsage) {
    kSecKeyUsageAll              = 0x7FFFFFFF
};
#endif /* !_SECURITY_VERSION_GREATER_THAN_57610_ */
extern const void * kSecCSRChallengePassword;
extern const void * kSecSubjectAltName;
extern const void * kSecCertificateKeyUsage;
extern const void * kSecCSRBasicContraintsPathLen;
extern const void * kSecCertificateExtensions;
extern const void * kSecCertificateExtensionsEncoded;
extern const void * kSecOidCommonName;
extern const void * kSecOidCountryName;
extern const void * kSecOidStateProvinceName;
extern const void * kSecOidLocalityName;
extern const void * kSecOidOrganization;
extern const void * kSecOidOrganizationalUnit;
extern bool SecCertificateIsValid(SecCertificateRef certificate, CFAbsoluteTime verifyTime);
extern CFAbsoluteTime SecCertificateNotValidAfter(SecCertificateRef certificate);
extern SecCertificateRef SecGenerateSelfSignedCertificate(CFArrayRef subject, CFDictionaryRef parameters, SecKeyRef publicKey, SecKeyRef privateKey);
extern SecIdentityRef SecIdentityCreate(CFAllocatorRef allocator, SecCertificateRef certificate, SecKeyRef privateKey);
