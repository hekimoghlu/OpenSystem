/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 1, 2022.
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
//  LWCRHelper.h
//  Security
//

#ifndef LWCRHelper_h
#define LWCRHelper_h

#import <CoreFoundation/CoreFoundation.h>
#import <TargetConditionals.h>
#include "requirement.h"

__BEGIN_DECLS

extern const uint8_t platformReqData[];
extern const size_t platformReqDataLen;

extern const uint8_t testflightReqData[];
extern const size_t testflightReqDataLen;
extern const uint8_t developmentReqData[];
extern const size_t developmentReqDataLen;
extern const uint8_t appStoreReqData[];
extern const size_t appStoreReqDataLen;
extern const uint8_t developerIDReqData[];
extern const size_t developerIDReqDataLen;

CFDictionaryRef copyDefaultDesignatedLWCRMaker(unsigned int validationCategory, const char* signingIdentifier, const char* teamIdentifier, CFArrayRef allCdhashes);

#if !TARGET_OS_SIMULATOR
OSStatus validateLightweightCodeRequirementData(CFDataRef lwcrData);
bool evaluateLightweightCodeRequirement(const Security::CodeSigning::Requirement::Context &ctx, CFDataRef lwcrData);
void evaluateLightweightCodeRequirementInKernel(audit_token_t token, CFDataRef lwcrData);
#endif

__END_DECLS
#endif /* LWCRHelper_h */
