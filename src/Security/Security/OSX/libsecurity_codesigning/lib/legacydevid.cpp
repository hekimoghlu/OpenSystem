/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 22, 2022.
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
#include "legacydevid.h"
#include "SecAssessment.h"
#include "requirement.h"

#include <Security/SecCertificatePriv.h>

namespace Security {
namespace CodeSigning {

static const CFStringRef kLegacyPolicyPreferenceDomain = CFSTR("com.apple.security.syspolicy");
static const CFStringRef kLegacyPolicyAccountCreationCutOff = CFSTR("AccountCreationCutOffDate");
static const CFStringRef kLegacyPolicySecureTimestampCutOff = CFSTR("SecureTimestampCutOffDate");
static const CFAbsoluteTime kLegacyPolicyAccountCreationDefaultCutOff = 576374400.0; // seconds from January 1, 2001 to April 7, 2019 GMT
static const CFAbsoluteTime kLegacyPolicySecureTimestampDefaultCutOff = 581040000.0; // seconds from January 1, 2001 to June 1, 2019 GMT

static CFDateRef
copyCutOffDate(const CFStringRef key, CFAbsoluteTime defaultCutoff)
{
    CFDateRef defaultDate = CFDateCreate(NULL, defaultCutoff);
    CFDateRef outputDate = defaultDate;
    CFDateRef prefDate = NULL;

    CFTypeRef prefVal = (CFDateRef)CFPreferencesCopyValue(key,
                                                          kLegacyPolicyPreferenceDomain,
                                                          kCFPreferencesCurrentUser,
                                                          kCFPreferencesAnyHost);
    if (prefVal && CFGetTypeID(prefVal) == CFDateGetTypeID()) {
        prefDate = (CFDateRef)prefVal;
    }

    if (prefDate) {
        CFComparisonResult res = CFDateCompare(defaultDate, prefDate, NULL);
        if (res > 0) {
            outputDate = prefDate;
        }
    }

    CFRetain(outputDate);

    if (prefVal) {
        CFRelease(prefVal);
    }
    if (defaultDate) {
        CFRelease(defaultDate);
    }
    return outputDate;
}

bool
meetsDeveloperIDLegacyAllowedPolicy(const Requirement::Context *context)
{
    CFRef<CFDataRef> cd;
    CFRef<CFErrorRef> error;
    CFRef<CFStringRef> teamID;
    bool meets_legacy_policy = false;
    SecCSDigestAlgorithm hashType = kSecCodeSignatureNoHash;
    SecCertificateRef cert = NULL;
    CFAbsoluteTime accountCreationTime = 0.0;

    if (context == NULL) {
        meets_legacy_policy = false;
        goto lb_exit;
    }

    // First check account creation date in certs
    // An account creation date after the cut off must be notarized so it fails the legacy policy.
    // No account creation date or an account creation date before the cut off requires additional checking
    cert = context->cert(Requirement::leafCert);
    if (SecCertificateGetDeveloperIDDate(cert, &accountCreationTime, &error.aref())) {
        //There is an account creation date
        CFRef<CFDateRef> accountCreationDate = CFDateCreate(NULL, accountCreationTime);
        CFRef<CFDateRef> accountCreationCutoffDate = copyCutOffDate(kLegacyPolicyAccountCreationCutOff,
                                                                    kLegacyPolicyAccountCreationDefaultCutOff);
        secinfo("meetsDeveloperIDLegacyAllowedPolicy", "Account Creation Date Cutoff: %@", accountCreationCutoffDate.get());
        secinfo("meetsDeveloperIDLegacyAllowedPolicy", "Account Creation date: %@", accountCreationDate.get());

        CFComparisonResult res = CFDateCompare(accountCreationDate, accountCreationCutoffDate, NULL);
        if (res >= 0) {
            // The account was created on or after our cut off so it doesn't meet legacy policy
            meets_legacy_policy = false;
            secinfo("meetsDeveloperIDLegacyAllowedPolicy", "Account creation date %@ is after cut-off %@", accountCreationDate.get(), accountCreationCutoffDate.get());
            goto lb_exit;
        }
        // Account creation date before the cut off means we fall through
    } else {
        CFIndex errorCode = CFErrorGetCode(error);
        if (errorCode != errSecMissingRequiredExtension) {
            secerror("Unexpected error checking account creation date: %ld", errorCode);
            meets_legacy_policy = false;
            goto lb_exit;
        }
        // there was no account creation date so fall through
    }

    // Next check secure time stamp
    if (context->secureTimestamp) {
        CFRef<CFDateRef> secureTimestampCutoffDate = copyCutOffDate(kLegacyPolicySecureTimestampCutOff,
                                                                    kLegacyPolicySecureTimestampDefaultCutOff);
        secinfo("meetsDeveloperIDLegacyAllowedPolicy", "Secure Timestamp Cutoff Date cutoff: %@", secureTimestampCutoffDate.get());
        secinfo("meetsDevleoperIDLegacyAllowedPolicy", "Secure Timestamp: %@", context->secureTimestamp);
        CFComparisonResult res = CFDateCompare(context->secureTimestamp, secureTimestampCutoffDate, NULL);
        if (res >= 0) {
            // Secure timestamp is on or after the cut of so it doesn't meet legacy policy
            meets_legacy_policy = false;
            secinfo("meetsDeveloperIDLegacyAllowedPolicy", "Secure timestamp %@ is after cut-off %@", context->secureTimestamp, secureTimestampCutoffDate.get());
        } else {
            // Secure timestamp is before the cut off so we meet the legacy policy
            meets_legacy_policy = true;
        }
    }

    if (!meets_legacy_policy) {
        // Just check against the legacy lists, both by hash and team ID.
        if (context->directory) {
            cd.take(context->directory->cdhash());
            hashType = (SecCSDigestAlgorithm)context->directory->hashType;
        } else if (context->packageChecksum) {
            cd = context->packageChecksum;
            hashType = context->packageAlgorithm;
        }

        if (cd.get() == NULL) {
            // No cdhash means we can't check the legacy lists
            meets_legacy_policy = false;
            goto lb_exit;
        }

        if (context->teamIdentifier) {
            teamID.take(CFStringCreateWithCString(kCFAllocatorDefault, context->teamIdentifier, kCFStringEncodingUTF8));
        }

        secnotice("legacy_list", "checking the legacy list for %d, %@, %@", hashType, cd.get(), teamID.get());
    #if TARGET_OS_OSX
        if (SecAssessmentLegacyCheck(cd, hashType, teamID, &error.aref())) {
            meets_legacy_policy = true;
        } else {
            meets_legacy_policy = false;
            if (error.get() != NULL) {
                secerror("Error checking with notarization daemon: %ld", CFErrorGetCode(error));
            }
        }
    #endif
    }
lb_exit:
    secnotice("legacy_list", "meetsDeveloperIDLegacyAllowedPolicy = %d", meets_legacy_policy);
    return meets_legacy_policy;
}

}
}
