/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 4, 2025.
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
#pragma once

#if USE(APPLE_INTERNAL_SDK)

#include <Security/SecAccessControlPriv.h>
#include <Security/SecCertificatePriv.h>
#include <Security/SecCode.h>
#include <Security/SecCodePriv.h>
#include <Security/SecIdentityPriv.h>
#include <Security/SecItemPriv.h>
#include <Security/SecKeyPriv.h>
#include <Security/SecStaticCode.h>
#include <Security/SecTask.h>
#include <Security/SecTrustPriv.h>

#if PLATFORM(MAC)
#include <Security/keyTemplates.h>
#endif

#else

#include <CoreFoundation/CoreFoundation.h>
#include <Security/SecBase.h>

#if __has_include(<Security/CSCommon.h>)
#include <Security/CSCommon.h>
#endif

typedef uint32_t SecSignatureHashAlgorithm;
enum {
    kSecSignatureHashAlgorithmUnknown = 0,
    kSecSignatureHashAlgorithmMD2 = 1,
    kSecSignatureHashAlgorithmMD4 = 2,
    kSecSignatureHashAlgorithmMD5 = 3,
    kSecSignatureHashAlgorithmSHA1 = 4,
    kSecSignatureHashAlgorithmSHA224 = 5,
    kSecSignatureHashAlgorithmSHA256 = 6,
    kSecSignatureHashAlgorithmSHA384 = 7,
    kSecSignatureHashAlgorithmSHA512 = 8
};

WTF_EXTERN_C_BEGIN

#if !__has_include(<Security/CSCommon.h>)
typedef struct __SecCode const *SecStaticCodeRef;

typedef uint32_t SecCSFlags;
enum {
    kSecCSDefaultFlags = 0,
};
#endif

#if PLATFORM(IOS_FAMILY)
extern const CFStringRef kSecCodeInfoUnique;

OSStatus SecStaticCodeCreateWithPath(CFURLRef, SecCSFlags, SecStaticCodeRef * CF_RETURNS_RETAINED);
OSStatus SecCodeCopySigningInformation(SecStaticCodeRef, SecCSFlags, CFDictionaryRef * CF_RETURNS_RETAINED);
#endif

#if PLATFORM(MAC)
OSStatus SecTrustedApplicationCreateFromPath(const char* path, SecTrustedApplicationRef*);
#endif

SecSignatureHashAlgorithm SecCertificateGetSignatureHashAlgorithm(SecCertificateRef);
extern const CFStringRef kSecAttrNoLegacy;

extern const CFStringRef kSecAttrAlias;

WTF_EXTERN_C_END

#endif // USE(APPLE_INTERNAL_SDK)

typedef struct __SecTask *SecTaskRef;
typedef struct __SecTrust *SecTrustRef;

WTF_EXTERN_C_BEGIN

SecTaskRef SecTaskCreateWithAuditToken(CFAllocatorRef, audit_token_t);
SecTaskRef SecTaskCreateFromSelf(CFAllocatorRef);
CFStringRef SecTaskCopySigningIdentifier(SecTaskRef, CFErrorRef *);
CFTypeRef SecTaskCopyValueForEntitlement(SecTaskRef, CFStringRef entitlement, CFErrorRef*);
uint32_t SecTaskGetCodeSignStatus(SecTaskRef);
SecIdentityRef SecIdentityCreate(CFAllocatorRef, SecCertificateRef, SecKeyRef);
SecAccessControlRef SecAccessControlCreateFromData(CFAllocatorRef, CFDataRef, CFErrorRef*);
CFDataRef SecAccessControlCopyData(SecAccessControlRef);

CFDataRef SecKeyCopySubjectPublicKeyInfo(SecKeyRef);

OSStatus SecCodeValidateFileResource(SecStaticCodeRef, CFStringRef, CFDataRef, SecCSFlags);

#if PLATFORM(MAC)
#include <Security/SecAsn1Types.h>
ALLOW_DEPRECATED_DECLARATIONS_BEGIN
extern const SecAsn1Template kSecAsn1AlgorithmIDTemplate[];
extern const SecAsn1Template kSecAsn1SubjectPublicKeyInfoTemplate[];
ALLOW_DEPRECATED_DECLARATIONS_END
#endif

#if PLATFORM(COCOA)
CF_RETURNS_RETAINED CFDataRef SecTrustSerialize(SecTrustRef, CFErrorRef *);
CF_RETURNS_RETAINED SecTrustRef SecTrustDeserialize(CFDataRef serializedTrust, CFErrorRef *);
CF_RETURNS_RETAINED CFPropertyListRef SecTrustCopyPropertyListRepresentation(SecTrustRef, CFErrorRef *);
CF_RETURNS_RETAINED SecTrustRef SecTrustCreateFromPropertyListRepresentation(CFPropertyListRef trustPlist, CFErrorRef *);
#endif

CF_RETURNS_RETAINED CFDictionaryRef SecTrustCopyInfo(SecTrustRef);

OSStatus SecTrustSetClientAuditToken(SecTrustRef, CFDataRef);

extern const CFStringRef kSecTrustInfoExtendedValidationKey;
extern const CFStringRef kSecTrustInfoCompanyNameKey;
extern const CFStringRef kSecTrustInfoRevocationKey;

WTF_EXTERN_C_END
