/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 11, 2022.
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
#import "SynchronousLoaderClient.h"

#import "AuthenticationChallenge.h"

namespace WebCore {

void SynchronousLoaderClient::didReceiveAuthenticationChallenge(ResourceHandle*, const AuthenticationChallenge& challenge)
{
    // FIXME: The user should be asked for credentials, as in async case.
    [challenge.sender() continueWithoutCredentialForAuthenticationChallenge:challenge.nsURLAuthenticationChallenge()];
}

ResourceError SynchronousLoaderClient::platformBadResponseError()
{
    RetainPtr<NSError> error = adoptNS([[NSError alloc] initWithDomain:NSURLErrorDomain code:NSURLErrorBadServerResponse userInfo:nil]);
    return error.get();
}

}
