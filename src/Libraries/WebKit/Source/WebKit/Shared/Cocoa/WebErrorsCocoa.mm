/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 6, 2023.
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
#import "WebErrors.h"

#import "APIError.h"
#import "WKErrorRef.h"
#import <WebCore/LocalizedStrings.h>
#import <WebCore/ResourceRequest.h>
#import <WebCore/ResourceResponse.h>

namespace WebKit {
using namespace WebCore;

static RetainPtr<NSError> createNSError(NSString* domain, int code, NSURL *URL)
{
    NSDictionary *userInfo = [NSDictionary dictionaryWithObjectsAndKeys:
        URL, @"NSErrorFailingURLKey",
        [URL absoluteString], @"NSErrorFailingURLStringKey",
        nil];

    return adoptNS([[NSError alloc] initWithDomain:domain code:code userInfo:userInfo]);
}

ResourceError cancelledError(const ResourceRequest& request)
{
    return ResourceError(createNSError(NSURLErrorDomain, NSURLErrorCancelled, request.url()).get());
}

ResourceError fileDoesNotExistError(const ResourceResponse& response)
{
    return ResourceError(createNSError(NSURLErrorDomain, NSURLErrorFileDoesNotExist, response.url()).get());
}

ResourceError decodeError(const URL& url)
{
    return ResourceError(createNSError(NSURLErrorDomain, NSURLErrorCannotDecodeContentData, url).get());
}

} // namespace WebKit
