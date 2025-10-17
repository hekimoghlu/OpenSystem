/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 27, 2023.
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
#import "config.h"
#import "AuthenticationChallengeDispositionCocoa.h"

namespace WebKit {

AuthenticationChallengeDisposition toAuthenticationChallengeDisposition(NSURLSessionAuthChallengeDisposition disposition)
{
    switch (disposition) {
    case NSURLSessionAuthChallengeUseCredential:
        return AuthenticationChallengeDisposition::UseCredential;
    case NSURLSessionAuthChallengePerformDefaultHandling:
        return AuthenticationChallengeDisposition::PerformDefaultHandling;
    case NSURLSessionAuthChallengeCancelAuthenticationChallenge:
        return AuthenticationChallengeDisposition::Cancel;
    case NSURLSessionAuthChallengeRejectProtectionSpace:
        return AuthenticationChallengeDisposition::RejectProtectionSpaceAndContinue;
    }
    [NSException raise:NSInvalidArgumentException format:@"Invalid NSURLSessionAuthChallengeDisposition (%ld)", (long)disposition];
}

NSURLSessionAuthChallengeDisposition fromAuthenticationChallengeDisposition(AuthenticationChallengeDisposition disposition)
{
    switch (disposition) {
    case AuthenticationChallengeDisposition::UseCredential:
        return NSURLSessionAuthChallengeUseCredential;
    case AuthenticationChallengeDisposition::PerformDefaultHandling:
        return NSURLSessionAuthChallengePerformDefaultHandling;
    case AuthenticationChallengeDisposition::Cancel:
        return NSURLSessionAuthChallengeCancelAuthenticationChallenge;
    case AuthenticationChallengeDisposition::RejectProtectionSpaceAndContinue:
        return NSURLSessionAuthChallengeRejectProtectionSpace;
    }
    [NSException raise:NSInvalidArgumentException format:@"Invalid AuthenticationChallengeDisposition (%ld)", (long)disposition];
}

} // namespace WebKit
