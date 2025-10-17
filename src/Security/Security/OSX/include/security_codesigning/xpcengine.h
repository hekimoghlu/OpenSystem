/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 22, 2023.
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
#ifndef _H_XPCENGINE
#define _H_XPCENGINE

#include "SecAssessment.h"
#include "policydb.h"
#include <xpc/private.h>
#include <Security/Security.h>

namespace Security {
namespace CodeSigning {


void xpcEngineAssess(CFURLRef path, SecAssessmentFlags flags, CFDictionaryRef context, CFMutableDictionaryRef result);
CFDictionaryRef xpcEngineUpdate(CFTypeRef target, SecAssessmentFlags flags, CFDictionaryRef context)
    CF_RETURNS_RETAINED;
bool xpcEngineControl(const char *name);
void xpcEngineRecord(CFDictionaryRef info);
void xpcEngineCheckDevID(CFBooleanRef* result);
void xpcEngineCheckNotarized(CFBooleanRef* result);

void xpcEngineTicketRegister(CFDataRef ticketData);
void xpcEngineTicketLookup(CFDataRef hashData, SecCSDigestAlgorithm hashType, SecAssessmentTicketFlags flags, double *date);
void xpcEngineLegacyCheck(CFDataRef hashData, SecCSDigestAlgorithm hashType, CFStringRef teamID);
void xpcEngineEnable(void);
void xpcEngineDisable(void);

} // end namespace CodeSigning
} // end namespace Security

#endif //_H_XPCENGINE
