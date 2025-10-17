/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 6, 2022.
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
#ifndef _H_SECASSESSMENT_PRIV
#define _H_SECASSESSMENT_PRIV

#include <CoreFoundation/CoreFoundation.h>
#include <Security/SecAssessment.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct SecAssessmentFunctions_t {
    Boolean (*ticketRegistration)(CFDataRef, CFErrorRef *);
    Boolean (*ticketLookup)(CFDataRef, SecCSDigestAlgorithm, SecAssessmentTicketFlags, double *, CFErrorRef *);
    Boolean (*legacyCheck)(CFDataRef, SecCSDigestAlgorithm, CFStringRef, CFErrorRef *);
} SecAssessmentFunctions_t;

/// Registers local function handlers, only for use in syspolicyd to prevent self XPC.
Boolean SecAssessmentRegisterFunctions(SecAssessmentFunctions_t *overrides);

#ifdef __cplusplus
}
#endif

#endif //_H_SECASSESSMENT_PRIV
