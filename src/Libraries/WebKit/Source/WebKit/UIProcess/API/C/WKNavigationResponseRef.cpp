/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 25, 2022.
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
#include "WKNavigationResponseRef.h"

#include "APINavigationResponse.h"
#include "WKAPICast.h"

WKTypeID WKNavigationResponseGetTypeID()
{
    return WebKit::toAPI(API::NavigationResponse::APIType);
}

bool WKNavigationResponseCanShowMIMEType(WKNavigationResponseRef response)
{
    return WebKit::toImpl(response)->canShowMIMEType();
}

WKURLResponseRef WKNavigationResponseCopyResponse(WKNavigationResponseRef response)
{
    return WebKit::toAPI(API::URLResponse::create(WebKit::toImpl(response)->response()).leakRef());
}

WKFrameInfoRef WKNavigationResponseCopyFrameInfo(WKNavigationResponseRef response)
{
    Ref frame = WebKit::toImpl(response)->frame();
    return WebKit::toAPI(&frame.leakRef());
}
