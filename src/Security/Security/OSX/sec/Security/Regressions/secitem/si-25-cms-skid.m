/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 30, 2025.
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
#include "shared_regressions.h"

#import <AssertMacros.h>
#import <Foundation/Foundation.h>

#import <Security/SecCMS.h>
#import <Security/SecTrust.h>
#import <Security/SecTrustPriv.h>
#include <utilities/SecCFRelease.h>

#import "si-25-cms-skid.h"

static void test_cms_verification(void)
{
    NSData *content = [NSData dataWithBytes:_content length:sizeof(_content)];
    NSData *signedData = [NSData dataWithBytes:_signedData length:sizeof(_signedData)];

    SecPolicyRef policy = SecPolicyCreateBasicX509();
    SecTrustRef trust = NULL;
    CFArrayRef certificates = NULL;

    ok_status(SecCMSVerify((__bridge CFDataRef)signedData, (__bridge CFDataRef)content, policy, &trust, NULL), "verify CMS message");

    /* verify that CMS stack found the certs in the CMS (using the SKID) and stuck them in the trust ref */
    ok_status(SecTrustCopyInputCertificates(trust, &certificates), "copy input certificates");
    CFIndex expectedCertCount = 4;
    is(CFArrayGetCount(certificates), expectedCertCount, "%d certs in the cms", (int)expectedCertCount);

    CFReleaseNull(policy);
    CFReleaseNull(trust);
    CFReleaseNull(certificates);
}

int si_25_cms_skid(int argc, char *const *argv)
{
    plan_tests(3);

    test_cms_verification();

    return 0;
}
