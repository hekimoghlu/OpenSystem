/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 24, 2021.
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
#include <sys/cdefs.h>
#include <AssertMacros.h>

#include <utilities/SecCFRelease.h>

#include <Security/SecBase.h>
#include <Security/SecImportExport.h>
#include <Security/SecKeychain.h>
#include <Security/SecCertificatePriv.h>
#include <Security/SecTrustSettings.h>
#include <Security/SecItem.h>
#include <Security/SecTrust.h>
#include <Security/SecPolicy.h>
#include <Security/CMSDecoder.h>

#define kSystemLoginKeychainPath "/Library/Keychains/System.keychain"

#include "regressions/test/testmore.h"
#include "cms_regressions.h"
#include "cms-trust-settings-test.h"

// See <rdar://problem/8115188>
static void test(void) {
    SecCertificateRef cert = NULL;
    SecKeychainRef kcRef = NULL;
    CFMutableDictionaryRef query = NULL;
    CFDictionaryRef trustSettings = NULL;
    CFArrayRef persistentRef = NULL;
    CMSDecoderRef decoder = NULL;
    SecPolicyRef policy = NULL;
    SecTrustRef trust = NULL;
    CMSSignerStatus signerStatus = kCMSSignerInvalidIndex;
    SecTrustResultType trustResult = kSecTrustResultInvalid;

    /* Add cert to keychain */
    ok(cert = SecCertificateCreateWithBytes(NULL, _cert, sizeof(_cert)), "Create cert");
    ok_status(SecKeychainOpen(kSystemLoginKeychainPath, &kcRef), "Open system keychain");
    if (!kcRef) {
        goto out;
    }
    ok(query = CFDictionaryCreateMutable(NULL, 3, &kCFTypeDictionaryKeyCallBacks,
                                         &kCFTypeDictionaryValueCallBacks),
       "Create SecItem dictionary");
    CFDictionaryAddValue(query, kSecValueRef, cert);
    CFDictionaryAddValue(query, kSecUseKeychain, kcRef);
    CFDictionaryAddValue(query, kSecReturnPersistentRef, kCFBooleanTrue);
    ok_status(SecItemAdd(query, (void *)&persistentRef),
              "Add cert to system keychain");

    /* Set trust settings */
    CFStringRef temp = kSecTrustSettingsResult;
    uint32_t otherTemp = kSecTrustSettingsResultDeny;
    CFNumberRef deny = CFNumberCreate(NULL, kCFNumberSInt32Type, &otherTemp);
    trustSettings = CFDictionaryCreate(NULL, (const void **)&temp, (const void **)&deny, 1,
                                       &kCFTypeDictionaryKeyCallBacks, &kCFTypeDictionaryValueCallBacks);
    CFReleaseNull(deny);
    ok_status(SecTrustSettingsSetTrustSettings(cert, kSecTrustSettingsDomainAdmin, trustSettings),
              "Set cert as denied");
    // Wait for trustd to get the message
    sleep(1);

    /* Create the Decoder */
    ok_status(CMSDecoderCreate(&decoder), "Create CMS decoder");
    ok_status(CMSDecoderUpdateMessage(decoder, _signed_message, sizeof(_signed_message)),
              "Update decoder with CMS message");
    ok_status(CMSDecoderFinalizeMessage(decoder), "Finalize decoder");

    /* Evaluate trust */
    ok(policy = SecPolicyCreateBasicX509(), "Create policy");
    ok_status(CMSDecoderCopySignerStatus(decoder, 0, policy, true, &signerStatus, &trust, NULL),
              "Copy Signer status");
    ok_status(SecTrustGetTrustResult(trust, &trustResult), "Get trust result");
    is(trustResult, kSecTrustResultDeny, "Not denied");

out:
    if (persistentRef) {
        CFTypeRef item = CFArrayGetValueAtIndex(persistentRef, 0);
        CFDictionaryRef del = CFDictionaryCreate(NULL, (const void **)&kSecValuePersistentRef, &item, 1,
                                   &kCFTypeDictionaryKeyCallBacks, &kCFTypeDictionaryValueCallBacks);
        SecItemDelete(del);
        CFReleaseNull(del);
    }
    CFReleaseNull(cert);
    CFReleaseNull(kcRef);
    CFReleaseNull(query);
    CFReleaseNull(persistentRef);
    CFReleaseNull(trustSettings);
    CFReleaseNull(decoder);
    CFReleaseNull(policy);
    CFReleaseNull(trust);
}

int cms_trust_settings_test(int argc, char *const *argv) {
    plan_tests(12);

#if !TARGET_OS_IPHONE
    if (getuid() != 0) {
        printf("Test must be run as root on OS X");
        return 0;
    }
#endif

    test();

    return 0;
}
