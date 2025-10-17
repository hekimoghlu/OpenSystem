/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 5, 2024.
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
//  SecCertificateFuzzer.c
//  Security
//

#include <CoreFoundation/CoreFoundation.h>
#include <Security/Security.h>
#include <Security/SecCertificatePriv.h>
#include "SecCertificateFuzzer.h"

extern bool SecCertificateHasOCSPNoCheckMarkerExtension(SecCertificateRef certificate);

int
SecCertificateFuzzer(const void *data, size_t len)
{
    CFDataRef d = CFDataCreateWithBytesNoCopy(NULL, data, len, kCFAllocatorNull);
    if (d) {
        SecCertificateRef cert = SecCertificateCreateWithData(NULL, d);
        CFRelease(d);
        if (cert) {
            CFStringRef summary = SecCertificateCopySubjectSummary(cert);
            if (summary) {
                CFRelease(summary);
            }
            CFArrayRef properties = SecCertificateCopyProperties(cert);
            if (properties) {
                CFRelease(properties);
            }
            CFArrayRef country = SecCertificateCopyCountry(cert);
            if (country) {
                CFRelease(country);
            }
            CFStringRef subject = SecCertificateCopySubjectString(cert);
            if (subject) {
                CFRelease(subject);
            }
            CFDataRef issuer = SecCertificateCopyIssuerSequence(cert);
            if (issuer) {
                CFRelease(issuer);
            }
            CFDataRef precert = SecCertificateCopyPrecertTBS(cert);
            if (precert) {
                CFRelease(precert);
            }
            (void)SecCertificateHasOCSPNoCheckMarkerExtension(cert);
            CFRelease(cert);
        }
    }

    return 0;
}
