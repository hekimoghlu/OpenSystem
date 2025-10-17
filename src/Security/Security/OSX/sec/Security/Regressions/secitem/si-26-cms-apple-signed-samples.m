/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 7, 2021.
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

//
//  si-26-cms-apple-signed-samples.m
//  SharedRegressions
//
//

#import <Foundation/Foundation.h>
#include <Security/Security.h>
#include <Security/SecCertificatePriv.h>
#include <Security/SecPolicyPriv.h>
#include <Security/CMSDecoder.h>
#include <utilities/SecCFRelease.h>

#include "shared_regressions.h"
#include "si-26-cms-apple-signed-samples.h"

static void tests(void)
{
    SecPolicyRef policy = NULL;

    /* Create Mac Provisioning Profile policy instance. */
    isnt(policy = SecPolicyCreateOSXProvisioningProfileSigning(), NULL, "create policy");

    /* Verify signed content with this policy. */
    CMSDecoderRef decoder = NULL;
    CMSSignerStatus signerStatus = kCMSSignerInvalidIndex;
    OSStatus verifyResult = 0;
    ok_status(CMSDecoderCreate(&decoder),
              "create decoder");
    ok_status(CMSDecoderUpdateMessage(decoder, _TestProvisioningProfile, sizeof(_TestProvisioningProfile)),
              "update message");
    ok_status(CMSDecoderFinalizeMessage(decoder),
              "finalize message");
    ok_status(CMSDecoderCopySignerStatus(decoder, 0, policy, true, &signerStatus, NULL, &verifyResult),
              "copy signer status");
    is(signerStatus, kCMSSignerValid, "signer status valid");
    is(verifyResult, errSecSuccess, "verify result valid");

    CFReleaseSafe(decoder);
    CFReleaseSafe(policy);
}

int si_26_cms_apple_signed_samples(int argc, char *const *argv)
{
    plan_tests(7);

    tests();

    return 0;
}
