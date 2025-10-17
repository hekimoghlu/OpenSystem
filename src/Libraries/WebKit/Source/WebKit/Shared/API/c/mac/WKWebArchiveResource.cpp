/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 11, 2024.
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
#include "WKWebArchiveResource.h"

#include "APIData.h"
#include "APIWebArchiveResource.h"
#include "WKSharedAPICast.h"

WKTypeID WKWebArchiveResourceGetTypeID()
{
    return WebKit::toAPI(API::WebArchiveResource::APIType);
}

WKWebArchiveResourceRef WKWebArchiveResourceCreate(WKDataRef dataRef, WKURLRef URLRef, WKStringRef MIMETypeRef, WKStringRef textEncodingRef)
{
    auto webArchiveResource = API::WebArchiveResource::create(WebKit::toImpl(dataRef), WebKit::toWTFString(URLRef), WebKit::toWTFString(MIMETypeRef), WebKit::toWTFString(textEncodingRef));
    return WebKit::toAPI(&webArchiveResource.leakRef());
}

WKDataRef WKWebArchiveResourceCopyData(WKWebArchiveResourceRef webArchiveResourceRef)
{
    return WebKit::toAPI(&WebKit::toImpl(webArchiveResourceRef)->data().leakRef());
}

WKURLRef WKWebArchiveResourceCopyURL(WKWebArchiveResourceRef webArchiveResourceRef)
{
    return WebKit::toCopiedURLAPI(WebKit::toImpl(webArchiveResourceRef)->url());
}

WKStringRef WKWebArchiveResourceCopyMIMEType(WKWebArchiveResourceRef webArchiveResourceRef)
{
    return WebKit::toCopiedAPI(WebKit::toImpl(webArchiveResourceRef)->mimeType());
}

WKStringRef WKWebArchiveResourceCopyTextEncoding(WKWebArchiveResourceRef webArchiveResourceRef)
{
    return WebKit::toCopiedAPI(WebKit::toImpl(webArchiveResourceRef)->textEncoding());
}
