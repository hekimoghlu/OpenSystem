/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 31, 2023.
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
#ifndef WKAuthenticationChallenge_h
#define WKAuthenticationChallenge_h

#include <WebKit/WKBase.h>

#ifdef __cplusplus
extern "C" {
#endif

WK_EXPORT WKTypeID WKAuthenticationChallengeGetTypeID();

WK_EXPORT WKAuthenticationDecisionListenerRef WKAuthenticationChallengeGetDecisionListener(WKAuthenticationChallengeRef);
WK_EXPORT WKProtectionSpaceRef WKAuthenticationChallengeGetProtectionSpace(WKAuthenticationChallengeRef);
WK_EXPORT WKCredentialRef WKAuthenticationChallengeGetProposedCredential(WKAuthenticationChallengeRef);
WK_EXPORT int WKAuthenticationChallengeGetPreviousFailureCount(WKAuthenticationChallengeRef);

#ifdef __cplusplus
}
#endif

#endif // WKAuthenticationChallenge_h
