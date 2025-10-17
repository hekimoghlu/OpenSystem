/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 19, 2024.
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
#include "WKUserContentURLPattern.h"

#include "APIUserContentURLPattern.h"
#include "WKAPICast.h"
#include "WKString.h"

WKTypeID WKUserContentURLPatternGetTypeID()
{
    return WebKit::toAPI(API::UserContentURLPattern::APIType);
}

WKUserContentURLPatternRef WKUserContentURLPatternCreate(WKStringRef patternRef)
{
    return WebKit::toAPI(&API::UserContentURLPattern::create(WebKit::toImpl(patternRef)->string()).leakRef());
}

WKStringRef WKUserContentURLPatternCopyHost(WKUserContentURLPatternRef urlPatternRef)
{
    return WebKit::toCopiedAPI(WebKit::toImpl(urlPatternRef)->host());
}

WKStringRef WKUserContentURLPatternCopyScheme(WKUserContentURLPatternRef urlPatternRef)
{
    return WebKit::toCopiedAPI(WebKit::toImpl(urlPatternRef)->scheme());
}

bool WKUserContentURLPatternIsValid(WKUserContentURLPatternRef urlPatternRef)
{
    return WebKit::toImpl(urlPatternRef)->isValid();
}

bool WKUserContentURLPatternMatchesURL(WKUserContentURLPatternRef urlPatternRef, WKURLRef urlRef)
{
    return WebKit::toImpl(urlPatternRef)->matchesURL(WebKit::toWTFString(urlRef));
}

bool WKUserContentURLPatternMatchesSubdomains(WKUserContentURLPatternRef urlPatternRef)
{
    return WebKit::toImpl(urlPatternRef)->matchesSubdomains();
}
