/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 13, 2022.
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
#include <Security/Security.h>
#include <Security/SecTrustPriv.h>
#include <Security/SecPolicyPriv.h>
#include <Security/SecCertificatePriv.h>
#include <Security/SecInternal.h>
#include <Security/SecTrustStatusCodes.h>
#include <CoreFoundation/CoreFoundation.h>
#include <libDER/oids.h>

struct resultmap_entry_s {
    const CFStringRef checkstr;
    const int32_t resultcode;
};
typedef struct resultmap_entry_s resultmap_entry_t;

const resultmap_entry_t resultmap[] = {
#undef POLICYCHECKMACRO
#define POLICYCHECKMACRO(NAME, TRUSTRESULT, SUBTYPE, LEAFCHECK, PATHCHECK, LEAFONLY, PROPFAILURE, CSSMERR, OSSTATUS) \
{ CFSTR(#NAME), (int32_t)CSSMERR },
#include "SecPolicyChecks.list"
};

static bool SecTrustDetailsHaveEKULeafErrorOnly(CFArrayRef details)
{
    CFIndex ix, count = (details) ? CFArrayGetCount(details) : 0;
    bool hasDisqualifyingError = false;
    for (ix = 0; ix < count; ix++) {
        CFDictionaryRef detail = (CFDictionaryRef)CFArrayGetValueAtIndex(details, ix);
        if (ix == 0) { // Leaf
            if (CFDictionaryGetCount(detail) != 1 || // One error
                CFDictionaryGetValue(detail, kSecPolicyCheckExtendedKeyUsage) != kCFBooleanFalse) {
                hasDisqualifyingError = true;
                break;
            }
        } else {
            if (CFDictionaryGetCount(detail) > 0) { // No errors on other certs
                hasDisqualifyingError = true;
                break;
            }
        }
    }
    if (hasDisqualifyingError) {
        return false;
    }
    return true;
}

// Returns true if both of the following are true:
// - policy is Apple SW Update Signing
// - leaf certificate has the oidAppleExtendedKeyUsageCodeSigningDev EKU purpose
//
static bool SecTrustIsDevelopmentUpdateSigning(SecTrustRef trust)
{
    bool result = false;
    CFArrayRef policies = NULL; /* must release */
    SecPolicyRef policy = NULL; /* must release */
    CFArrayRef chain = NULL; /* must release */
    SecCertificateRef cert = NULL;
    CFArrayRef ekus = NULL; /* must release */
    CFDataRef eku = NULL; /* must release */
    const DERItem *oid = &oidAppleExtendedKeyUsageCodeSigningDev;

    /* Apple SW Update Signing policy check */
    if ((SecTrustCopyPolicies(trust, &policies) != errSecSuccess) ||
        ((policy = SecPolicyCreateAppleSWUpdateSigning()) == NULL) ||
        (!CFArrayContainsValue(policies, CFRangeMake(0, CFArrayGetCount(policies)), policy))) {
        goto exit;
    }

    /* Apple Code Signing Dev EKU check */
    chain = SecTrustCopyCertificateChain(trust);
    if ((chain == NULL) ||
        ((cert = (SecCertificateRef)CFArrayGetValueAtIndex(chain, 0)) == NULL) ||
        ((ekus = SecCertificateCopyExtendedKeyUsage(cert)) == NULL) ||
        (oid->length > LONG_MAX) ||
        ((eku = CFDataCreate(kCFAllocatorDefault, oid->data, (CFIndex)oid->length)) == NULL) ||
        (!CFArrayContainsValue(ekus, CFRangeMake(0, CFArrayGetCount(ekus)), eku))) {
        goto exit;
    }

    result = true;

exit:
    CFReleaseSafe(eku);
    CFReleaseSafe(ekus);
    CFReleaseSafe(policies);
    CFReleaseSafe(policy);
    CFReleaseSafe(chain);
    return result;
}

//
// Returns a malloced array of SInt32 values, with the length in numStatusCodes,
// for the certificate specified by chain index in the given SecTrustRef.
//
// To match legacy behavior, the array actually allocates one element more than the
// value of numStatusCodes; if the certificate is revoked, the additional element
// at the end contains the CrlReason value.
//
// Caller must free the returned pointer.
//
SInt32 *SecTrustCopyStatusCodes(SecTrustRef trust,
    CFIndex index, CFIndex *numStatusCodes)
{
    if (!trust || !numStatusCodes) {
        return NULL;
    }
    *numStatusCodes = 0;
    CFArrayRef details = SecTrustCopyFilteredDetails(trust);
    CFIndex chainLength = (details) ? CFArrayGetCount(details) : 0;
    if (!(index < chainLength)) {
        CFReleaseSafe(details);
        return NULL;
    }
    CFDictionaryRef detail = (CFDictionaryRef)CFArrayGetValueAtIndex(details, index);
    CFIndex ix, detailCount = CFDictionaryGetCount(detail);
    if (detailCount < 0 || detailCount >= (long)((LONG_MAX / sizeof(SInt32)) - 1)) {
        CFReleaseSafe(details);
        return NULL;
    }
    *numStatusCodes = (unsigned int)detailCount;

    // Allocate one more entry than we need; this is used to store a CrlReason
    // at the end of the array.
    SInt32 *statusCodes = (SInt32*)malloc((size_t)(detailCount+1) * sizeof(SInt32));
    statusCodes[*numStatusCodes] = 0;

    const unsigned int resultmaplen = sizeof(resultmap) / sizeof(resultmap_entry_t);
    const void *keys[detailCount];
    CFDictionaryGetKeysAndValues(detail, &keys[0], NULL);
    for (ix = 0; ix < detailCount; ix++) {
        CFStringRef key = (CFStringRef)keys[ix];
        SInt32 statusCode = 0;
        for (unsigned int mapix = 0; mapix < resultmaplen; mapix++) {
            CFStringRef str = (CFStringRef) resultmap[mapix].checkstr;
            if (CFStringCompare(str, key, 0) == kCFCompareEqualTo) {
                statusCode = (SInt32) resultmap[mapix].resultcode;
                break;
            }
        }
        if (statusCode == (SInt32)0x80012407) {  /* CSSMERR_APPLETP_INVALID_EXTENDED_KEY_USAGE */
            // To match legacy behavior, we return a more specific result code if this is a
            // development signing certificate being evaluated for Apple SW Update Signing.
            // [27362805,41179903]
            if (index == 0 &&
                SecTrustIsDevelopmentUpdateSigning(trust) &&
                SecTrustDetailsHaveEKULeafErrorOnly(details)) {
                statusCode = (SInt32)0x80012433; /* CSSMERR_APPLETP_CODE_SIGN_DEVELOPMENT */
            }
        } else if (statusCode == (SInt32)0x8001210C) {  /* CSSMERR_TP_CERT_REVOKED */
            SInt32 reason;
            CFNumberRef number = (CFNumberRef)CFDictionaryGetValue(detail, key);
            if (number && CFNumberGetValue(number, kCFNumberSInt32Type, &reason)) {
                statusCodes[*numStatusCodes] = (SInt32)reason;
            }
        }
        statusCodes[ix] = statusCode;
    }

    CFReleaseSafe(details);
    return statusCodes;
}
