/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 10, 2025.
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
#include "config.h"
#include "WebErrors.h"

#include "APIError.h"
#include "Logging.h"
#include <WebCore/LocalizedStrings.h>
#include <WebCore/ResourceRequest.h>
#include <WebCore/ResourceResponse.h>

namespace WebKit {
using namespace WebCore;

ResourceError blockedError(const ResourceRequest& request)
{
    return ResourceError(API::Error::webKitPolicyErrorDomain(), API::Error::Policy::CannotUseRestrictedPort, request.url(), WEB_UI_STRING("Not allowed to use restricted network port", "WebKitErrorCannotUseRestrictedPort description"));
}

ResourceError blockedByContentBlockerError(const ResourceRequest& request)
{
    return ResourceError(API::Error::webKitPolicyErrorDomain(), API::Error::Policy::FrameLoadBlockedByContentBlocker, request.url(), WEB_UI_STRING("The URL was blocked by a content blocker", "WebKitErrorBlockedByContentBlocker description"));
}

ResourceError cannotShowURLError(const ResourceRequest& request)
{
    return ResourceError(API::Error::webKitPolicyErrorDomain(), API::Error::Policy::CannotShowURL, request.url(), WEB_UI_STRING("The URL canâ€™t be shown", "WebKitErrorCannotShowURL description"));
}

ResourceError wasBlockedByRestrictionsError(const ResourceRequest& request)
{
    return ResourceError(API::Error::webKitPolicyErrorDomain(), API::Error::Policy::FrameLoadBlockedByRestrictions, request.url(), WEB_UI_STRING("The URL was blocked by device restrictions", "WebKitErrorFrameLoadBlockedByRestrictions description"));
}

ResourceError interruptedForPolicyChangeError(const ResourceRequest& request)
{
    return ResourceError(API::Error::webKitPolicyErrorDomain(), API::Error::Policy::FrameLoadInterruptedByPolicyChange, request.url(), WEB_UI_STRING("Frame load interrupted", "WebKitErrorFrameLoadInterruptedByPolicyChange description"));
}

ResourceError ftpDisabledError(const ResourceRequest& request)
{
    return ResourceError(errorDomainWebKitInternal, 0, request.url(), "FTP URLs are disabled"_s, ResourceError::Type::AccessControl);
}

ResourceError failedCustomProtocolSyncLoad(const ResourceRequest& request)
{
    return ResourceError(errorDomainWebKitInternal, 0, request.url(), WEB_UI_STRING("Error handling synchronous load with custom protocol", "Custom protocol synchronous load failure description"));
}

#if ENABLE(CONTENT_FILTERING)
ResourceError blockedByContentFilterError(const ResourceRequest& request)
{
    return ResourceError(API::Error::webKitPolicyErrorDomain(), API::Error::Policy::FrameLoadBlockedByContentFilter, request.url(), WEB_UI_STRING("The URL was blocked by a content filter", "WebKitErrorFrameLoadBlockedByContentFilter description"));
}
#endif

ResourceError cannotShowMIMETypeError(const ResourceResponse& response)
{
    return ResourceError(API::Error::webKitPolicyErrorDomain(), API::Error::Policy::CannotShowMIMEType, response.url(), WEB_UI_STRING("Content with specified MIME type canâ€™t be shown", "WebKitErrorCannotShowMIMEType description"));
}

ResourceError pluginWillHandleLoadError(const ResourceResponse& response)
{
    return ResourceError(API::Error::webKitPluginErrorDomain(), API::Error::Plugin::PlugInWillHandleLoad, response.url(), WEB_UI_STRING("Plug-in handled load", "WebKitErrorPlugInWillHandleLoad description"));
}

#if !PLATFORM(COCOA)
ResourceError cancelledError(const ResourceRequest& request)
{
    return ResourceError(API::Error::webKitNetworkErrorDomain(), API::Error::Network::Cancelled, request.url(), WEB_UI_STRING("Load request cancelled", "Load request cancelled"));
}

ResourceError fileDoesNotExistError(const ResourceResponse& response)
{
    return ResourceError(API::Error::webKitNetworkErrorDomain(), API::Error::Network::FileDoesNotExist, response.url(), WEB_UI_STRING("File does not exist", "The requested file doesn't exist"));
}

ResourceError decodeError(const URL&)
{
    return { };
}
#endif

ResourceError httpsUpgradeRedirectLoopError(const ResourceRequest& request)
{
    return ResourceError(API::Error::webKitNetworkErrorDomain(), API::Error::Network::HTTPSUpgradeRedirectLoop, request.url(), WEB_UI_STRING("HTTPS Upgrade redirect loop detected", "WebKitErrorHTTPSUpgradeRedirectLoop description"));
}

ResourceError httpNavigationWithHTTPSOnlyError(const ResourceRequest& request)
{
    return ResourceError(API::Error::webKitNetworkErrorDomain(), API::Error::Network::HTTPNavigationWithHTTPSOnlyError, request.url(), WEB_UI_STRING("Navigation failed because the request was for an HTTP URL with HTTPS-Only enabled", "WebKitErrorHTTPSOnlyHTTPURL description"));
}

} // namespace WebKit
