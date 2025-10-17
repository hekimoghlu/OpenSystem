/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 1, 2022.
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
#include "WKURLResponse.h"

#include "APIURLResponse.h"
#include "WKAPICast.h"
#include <wtf/URL.h>

WKTypeID WKURLResponseGetTypeID()
{
    return WebKit::toAPI(API::URLResponse::APIType);
}

WKURLRef WKURLResponseCopyURL(WKURLResponseRef responseRef)
{
    return WebKit::toCopiedURLAPI(WebKit::toImpl(responseRef)->resourceResponse().url());
}

WKStringRef WKURLResponseCopyMIMEType(WKURLResponseRef responseRef)
{
    return WebKit::toCopiedAPI(WebKit::toImpl(responseRef)->resourceResponse().mimeType());
}

int32_t WKURLResponseHTTPStatusCode(WKURLResponseRef responseRef)
{
    return WebKit::toImpl(responseRef)->resourceResponse().httpStatusCode();
}

WKStringRef WKURLResponseCopySuggestedFilename(WKURLResponseRef responseRef)
{
    return WebKit::toCopiedAPI(WebKit::toImpl(responseRef)->resourceResponse().suggestedFilename());
}

bool WKURLResponseIsAttachment(WKURLResponseRef responseRef)
{
    return WebKit::toImpl(responseRef)->resourceResponse().isAttachment();
}

uint32_t WKURLResponseGetExpectedContentLength(WKURLResponseRef responseRef)
{
    return WebKit::toImpl(responseRef)->resourceResponse().expectedContentLength();
}
