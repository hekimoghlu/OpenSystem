/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 8, 2023.
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
#include "WKErrorRef.h"

#include "APIError.h"
#include "WKAPICast.h"

WKTypeID WKErrorGetTypeID()
{
    return WebKit::toAPI(API::Error::APIType);
}

WKStringRef WKErrorCopyWKErrorDomain()
{
    return WebKit::toCopiedAPI(API::Error::webKitErrorDomain());
}

WKStringRef WKErrorCopyDomain(WKErrorRef errorRef)
{
    return WebKit::toCopiedAPI(WebKit::toImpl(errorRef)->domain());
}

int WKErrorGetErrorCode(WKErrorRef errorRef)
{
    auto errorCode = WebKit::toImpl(errorRef)->errorCode();
    switch (errorCode) {
    case API::Error::Policy::CannotShowMIMEType:
        return kWKErrorCodeCannotShowMIMEType;
    case API::Error::Policy::CannotShowURL:
        return kWKErrorCodeCannotShowURL;
    case API::Error::Policy::FrameLoadInterruptedByPolicyChange:
        return kWKErrorCodeFrameLoadInterruptedByPolicyChange;
    case API::Error::Policy::CannotUseRestrictedPort:
        return kWKErrorCodeCannotUseRestrictedPort;
    case API::Error::Policy::FrameLoadBlockedByContentBlocker:
        return kWKErrorCodeFrameLoadBlockedByContentBlocker;
    case API::Error::Policy::FrameLoadBlockedByRestrictions:
        return kWKErrorCodeFrameLoadBlockedByRestrictions;
    case API::Error::Policy::FrameLoadBlockedByContentFilter:
        return kWKErrorCodeFrameLoadBlockedByContentFilter;
    case API::Error::Plugin::CannotFindPlugIn:
        return kWKErrorCodeCannotFindPlugIn;
    case API::Error::Plugin::CannotLoadPlugIn:
        return kWKErrorCodeCannotLoadPlugIn;
    case API::Error::Plugin::JavaUnavailable:
        return kWKErrorCodeJavaUnavailable;
    case API::Error::Plugin::PlugInCancelledConnection:
        return kWKErrorCodePlugInCancelledConnection;
    case API::Error::Plugin::PlugInWillHandleLoad:
        return kWKErrorCodePlugInWillHandleLoad;
    case API::Error::Plugin::InsecurePlugInVersion:
        return kWKErrorCodeInsecurePlugInVersion;
    case API::Error::General::Internal:
        return kWKErrorInternal;
    case API::Error::Network::Cancelled:
        return kWKErrorCodeCancelled;
    case API::Error::Network::FileDoesNotExist:
        return kWKErrorCodeFileDoesNotExist;
    }

    return errorCode;
}

WKURLRef WKErrorCopyFailingURL(WKErrorRef errorRef)
{
    return WebKit::toCopiedURLAPI(WebKit::toImpl(errorRef)->failingURL());
}

WKStringRef WKErrorCopyLocalizedDescription(WKErrorRef errorRef)
{
    return WebKit::toCopiedAPI(WebKit::toImpl(errorRef)->localizedDescription());
}
