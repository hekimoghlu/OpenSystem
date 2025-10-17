/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 2, 2022.
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
    @header SecTrustInternal
    This header provides the interface to internal functions used by SecTrust.
*/

#ifndef _SECURITY_SECTRUSTINTERNAL_H_
#define _SECURITY_SECTRUSTINTERNAL_H_

#include <Security/SecTrust.h>
#include <Security/SecTrustSettings.h>

__BEGIN_DECLS

/* args_in keys. */
#define kSecTrustCertificatesKey "certificates"
#define kSecTrustAnchorsKey "anchors"
#define kSecTrustAnchorsOnlyKey "anchorsOnly"
#define kSecTrustKeychainsAllowedKey "keychainsAllowed"
#define kSecTrustPoliciesKey "policies"
#define kSecTrustResponsesKey "responses"
#define kSecTrustSCTsKey "scts"
#define kSecTrustTrustedLogsKey "trustedLogs"
#define kSecTrustVerifyDateKey "verifyDate"
#define kSecTrustExceptionsKey "exceptions"
#define kSecTrustRevocationAdditionsKey "revocationCheck"
#define kSecTrustAuditTokenKey "auditToken"
#define kSecTrustSettingsAuthExternalForm "auth"
#define kSecTrustSettingsDomain "domain"
#define kSecTrustSettingsData "settings"
#define kSecTrustURLAttribution "attribution"

/* args_out keys. */
#define kSecTrustDetailsKey "details"
#define kSecTrustChainKey "chain"
#define kSecTrustResultKey "result"
#define kSecTrustInfoKey "info"

extern const CFStringRef kSecCertificateDetailSHA1Digest;

bool SecTrustIsTrustResultValid(SecTrustRef trust, CFAbsoluteTime verifyTime);

#if TARGET_OS_OSX
/* SecTrust functions */
SecKeyRef SecTrustCopyPublicKey_ios(SecTrustRef trust);
CFArrayRef SecTrustCopyProperties_ios(SecTrustRef trust);
#endif

/* SecTrustStore functions */
CFStringRef SecTrustSettingsDomainName(SecTrustSettingsDomain domain);
SecTrustSettingsDomain SecTrustSettingsDomainForName(CFStringRef domainName);
#if TARGET_OS_OSX
OSStatus SecTrustSettingsXPCRead(CFStringRef domain, CFDataRef *trustSettings);
OSStatus SecTrustSettingsXPCWrite(CFStringRef domain, CFDataRef auth, CFDataRef trustSettings);
#endif

#define kSecTrustEventNameKey "eventName"
#define kSecTrustEventAttributesKey "eventAttributes"
#define kSecTrustEventApplicationID "appID"

typedef enum {
    kSecTrustErrorSubTypeBlocked,
    kSecTrustErrorSubTypeRevoked,
    kSecTrustErrorSubTypeKeySize,
    kSecTrustErrorSubTypeWeakHash,
    kSecTrustErrorSubTypeDenied,
    kSecTrustErrorSubTypeCompliance,
    kSecTrustErrorSubTypePinning,
    kSecTrustErrorSubTypeTrust,
    kSecTrustErrorSubTypeUsage,
    kSecTrustErrorSubTypeName,
    kSecTrustErrorSubTypeExpired,
    kSecTrustErrorSubTypeInvalid,
} SecTrustErrorSubType;

#define __PC_SUBTYPE_   kSecTrustErrorSubTypeInvalid
#define __PC_SUBTYPE_N  kSecTrustErrorSubTypeName
#define __PC_SUBTYPE_E  kSecTrustErrorSubTypeExpired
#define __PC_SUBTYPE_S  kSecTrustErrorSubTypeKeySize
#define __PC_SUBTYPE_H  kSecTrustErrorSubTypeWeakHash
#define __PC_SUBTYPE_U  kSecTrustErrorSubTypeUsage
#define __PC_SUBTYPE_P  kSecTrustErrorSubTypePinning
#define __PC_SUBTYPE_V  kSecTrustErrorSubTypeRevoked
#define __PC_SUBTYPE_T  kSecTrustErrorSubTypeTrust
#define __PC_SUBTYPE_C  kSecTrustErrorSubTypeCompliance
#define __PC_SUBTYPE_D  kSecTrustErrorSubTypeDenied
#define __PC_SUBTYPE_B  kSecTrustErrorSubTypeBlocked

__END_DECLS

#endif /* !_SECURITY_SECTRUSTINTERNAL_H_ */
