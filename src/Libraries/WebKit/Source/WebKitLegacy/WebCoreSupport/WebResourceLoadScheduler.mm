/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 29, 2022.
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
#include "WebResourceLoadScheduler.h"

#include "WebKitErrorsPrivate.h"
#include <WebCore/ResourceError.h>
#include <WebCore/ResourceRequest.h>

WebCore::ResourceError WebResourceLoadScheduler::pluginWillHandleLoadError(const WebCore::ResourceResponse& response) const
{
    return WebResourceLoadScheduler::pluginWillHandleLoadErrorFromResponse(response);
}

WebCore::ResourceError WebResourceLoadScheduler::pluginWillHandleLoadErrorFromResponse(const WebCore::ResourceResponse& response)
{
    return [[[NSError alloc] _initWithPluginErrorCode:WebKitErrorPlugInWillHandleLoad contentURL:response.url() pluginPageURL:nil pluginName:nil MIMEType:response.mimeType()] autorelease];
}

WebCore::ResourceError WebResourceLoadScheduler::cancelledError(const WebCore::ResourceRequest& request) const
{
    return [NSError _webKitErrorWithDomain:NSURLErrorDomain code:NSURLErrorCancelled URL:request.url()];
}

WebCore::ResourceError WebResourceLoadScheduler::blockedError(const WebCore::ResourceRequest& request) const
{
    return WebResourceLoadScheduler::blockedErrorFromRequest(request);
}

WebCore::ResourceError WebResourceLoadScheduler::blockedErrorFromRequest(const WebCore::ResourceRequest& request)
{
    return [NSError _webKitErrorWithDomain:WebKitErrorDomain code:WebKitErrorCannotUseRestrictedPort URL:request.url()];
}

WebCore::ResourceError WebResourceLoadScheduler::blockedByContentBlockerError(const WebCore::ResourceRequest& request) const
{
    RELEASE_ASSERT_NOT_REACHED(); // Content blockers are not enabled in WebKit1.
}

WebCore::ResourceError WebResourceLoadScheduler::cannotShowURLError(const WebCore::ResourceRequest& request) const
{
    return [NSError _webKitErrorWithDomain:WebKitErrorDomain code:WebKitErrorCannotShowURL URL:request.url()];
}

WebCore::ResourceError WebResourceLoadScheduler::interruptedForPolicyChangeError(const WebCore::ResourceRequest& request) const
{
    return [NSError _webKitErrorWithDomain:WebKitErrorDomain code:WebKitErrorFrameLoadInterruptedByPolicyChange URL:request.url()];
}

#if ENABLE(CONTENT_FILTERING)
WebCore::ResourceError WebResourceLoadScheduler::blockedByContentFilterError(const WebCore::ResourceRequest& request) const
{
    return [NSError _webKitErrorWithDomain:WebKitErrorDomain code:WebKitErrorFrameLoadBlockedByContentFilter URL:request.url()];
}
#endif

WebCore::ResourceError WebResourceLoadScheduler::cannotShowMIMETypeError(const WebCore::ResourceResponse& response) const
{
    return [NSError _webKitErrorWithDomain:NSURLErrorDomain code:WebKitErrorCannotShowMIMEType URL:response.url()];
}

WebCore::ResourceError WebResourceLoadScheduler::fileDoesNotExistError(const WebCore::ResourceResponse& response) const
{
    return [NSError _webKitErrorWithDomain:NSURLErrorDomain code:NSURLErrorFileDoesNotExist URL:response.url()];
}

WebCore::ResourceError WebResourceLoadScheduler::httpsUpgradeRedirectLoopError(const WebCore::ResourceRequest&) const
{
    RELEASE_ASSERT_NOT_REACHED(); // This error should never be created in WebKit1 because HTTPSOnly/First aren't available.
}

WebCore::ResourceError WebResourceLoadScheduler::httpNavigationWithHTTPSOnlyError(const WebCore::ResourceRequest&) const
{
    RELEASE_ASSERT_NOT_REACHED(); // This error should never be created in WebKit1 because HTTPSOnly/First aren't available.
}
