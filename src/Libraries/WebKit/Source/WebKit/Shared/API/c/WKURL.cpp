/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 17, 2025.
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
#include "WKURL.h"

#include "WKAPICast.h"

WKTypeID WKURLGetTypeID()
{
    return WebKit::toAPI(API::URL::APIType);
}

WKURLRef WKURLCreateWithUTF8CString(const char* string)
{
    return WebKit::toAPI(&API::URL::create(String::fromUTF8(string)).leakRef());
}

WKURLRef WKURLCreateWithUTF8String(const char* string, size_t length)
{
    return WebKit::toAPI(&API::URL::create(String::fromUTF8(unsafeMakeSpan(string, length))).leakRef());
}

WKURLRef WKURLCreateWithBaseURL(WKURLRef baseURL, const char* relative)
{
    return WebKit::toAPI(&API::URL::create(WebKit::toImpl(baseURL), String::fromUTF8(relative)).leakRef());
}

WKStringRef WKURLCopyString(WKURLRef url)
{
    return WebKit::toCopiedAPI(WebKit::toImpl(url)->string());
}

bool WKURLIsEqual(WKURLRef a, WKURLRef b)
{
    return API::URL::equals(*WebKit::toImpl(a), *WebKit::toImpl(b));
}

WKStringRef WKURLCopyHostName(WKURLRef url)
{
    return WebKit::toCopiedAPI(WebKit::toImpl(url)->host());
}

WKStringRef WKURLCopyScheme(WKURLRef url)
{
    return WebKit::toCopiedAPI(WebKit::toImpl(url)->protocol());
}

WK_EXPORT WKStringRef WKURLCopyPath(WKURLRef url)
{
    return WebKit::toCopiedAPI(WebKit::toImpl(url)->path());
}

WKStringRef WKURLCopyLastPathComponent(WKURLRef url)
{
    return WebKit::toCopiedAPI(WebKit::toImpl(url)->lastPathComponent());
}
