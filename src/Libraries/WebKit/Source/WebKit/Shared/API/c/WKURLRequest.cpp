/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 30, 2024.
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
#include "WKURLRequest.h"

#include "APIURLRequest.h"
#include "WKAPICast.h"
#include "WKData.h"
#include <wtf/StdLibExtras.h>
#include <wtf/URL.h>

WKTypeID WKURLRequestGetTypeID()
{
    return WebKit::toAPI(API::URLRequest::APIType);
}

WKURLRequestRef WKURLRequestCreateWithWKURL(WKURLRef url)
{
    return WebKit::toAPI(&API::URLRequest::create(URL { WebKit::toImpl(url)->string() }).leakRef());
}

WKURLRef WKURLRequestCopyURL(WKURLRequestRef requestRef)
{
    return WebKit::toCopiedURLAPI(WebKit::toImpl(requestRef)->resourceRequest().url());
}

WKURLRef WKURLRequestCopyFirstPartyForCookies(WKURLRequestRef requestRef)
{
    return WebKit::toCopiedURLAPI(WebKit::toImpl(requestRef)->resourceRequest().firstPartyForCookies());
}

WKStringRef WKURLRequestCopyHTTPMethod(WKURLRequestRef requestRef)
{
    return WebKit::toCopiedAPI(WebKit::toImpl(requestRef)->resourceRequest().httpMethod());
}

WKURLRequestRef WKURLRequestCopySettingHTTPBody(WKURLRequestRef requestRef, WKDataRef body)
{
    WebCore::ResourceRequest requestCopy(WebKit::toImpl(requestRef)->resourceRequest());
    requestCopy.setHTTPBody(WebCore::FormData::create(WKDataGetSpan(body)));
    return WebKit::toAPI(&API::URLRequest::create(requestCopy).leakRef());
}

void WKURLRequestSetDefaultTimeoutInterval(double timeoutInterval)
{
    API::URLRequest::setDefaultTimeoutInterval(timeoutInterval);
}
