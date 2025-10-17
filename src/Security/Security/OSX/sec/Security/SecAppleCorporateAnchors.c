/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 9, 2023.
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
//  SecAppleCorporateAnchors.c
//  Security
//
//

#include <AssertMacros.h>
#include "SecAppleCorporateAnchors.h"
#include "AppleCorporateRootCertificates.h"
#include <Security/SecCertificatePriv.h>

// Assigns NULL to CF. Releases the value stored at CF unless it was NULL.  Always returns NULL, for your convenience
#define CFReleaseNull(CF) ({ __typeof__(CF) *const _pcf = &(CF), _cf = *_pcf; (_cf ? (*_pcf) = ((__typeof__(CF))0), (CFRelease(_cf), ((__typeof__(CF))0)) : _cf); })


// README: See AppleCorporateRootCertificates.h for instructions for adding new corporate roots
CFArrayRef SecCertificateCopyAppleCorporateRoots(void) {
    CFMutableArrayRef result = NULL;
    SecCertificateRef corp1 = NULL, corp2 = NULL, corp3 = NULL;

    require_quiet(corp1= SecCertificateCreateWithBytes(NULL, _AppleCorporateRootCA, sizeof(_AppleCorporateRootCA)),
                  errOut);
    require_quiet(corp2 = SecCertificateCreateWithBytes(NULL, _AppleCorporateRootCA2,
                                                                          sizeof(_AppleCorporateRootCA2)),
                  errOut);
    require_quiet(corp3 = SecCertificateCreateWithBytes(NULL, _AppleCorporateRootCA3,
                                                                          sizeof(_AppleCorporateRootCA3)),
                  errOut);

    require_quiet(result = CFArrayCreateMutable(NULL, 0, &kCFTypeArrayCallBacks), errOut);
    CFArrayAppendValue(result, corp1);
    CFArrayAppendValue(result, corp2);
    CFArrayAppendValue(result, corp3);

errOut:
    CFReleaseNull(corp1);
    CFReleaseNull(corp2);
    CFReleaseNull(corp3);
    return result;
}
