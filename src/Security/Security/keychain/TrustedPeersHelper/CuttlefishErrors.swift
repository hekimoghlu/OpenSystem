/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 31, 2022.
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

import Foundation

struct CuttlefishErrorMatcher {
    let code: CuttlefishErrorCode
}

// Use a 'pattern match operator' to make pretty case statements matching Cuttlefish errors
func ~= (pattern: CuttlefishErrorMatcher, value: Error?) -> Bool {
    guard let error = value else {
        return false
    }
    let nserror = error as NSError
    return nserror.isCuttlefishError(pattern.code)
}

// This function is only used by RetryingCKCodeService, which enforces a minimum
// retry interval of five seconds and a maximum time of 30 seconds. This means that
// -[NSError(Octagon) retryInterval], which defaults to 30 seconds, cannot be used.
// Instead, use -[NSError(Octagon) cuttlefishRetryAfter] to get the true value.
func CuttlefishRetryAfter(error: Error?) -> TimeInterval {
    guard let error = error else {
        return 0
    }
    let nserror = error as NSError
    return nserror.cuttlefishRetryAfter()
}
