/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 23, 2023.
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
#include "WKWebArchiveRef.h"

#include "APIArray.h"
#include "APIData.h"
#include "APIWebArchive.h"
#include "APIWebArchiveResource.h"
#include "InjectedBundleRangeHandle.h"
#include "WKBundleAPICast.h"
#include "WKSharedAPICast.h"
#include <WebCore/Range.h>
#include <WebCore/SimpleRange.h>

WKTypeID WKWebArchiveGetTypeID()
{
    return WebKit::toAPI(API::WebArchive::APIType);
}

WKWebArchiveRef WKWebArchiveCreate(WKWebArchiveResourceRef mainResourceRef, WKArrayRef subresourcesRef, WKArrayRef subframeArchivesRef)
{
    auto webArchive = API::WebArchive::create(WebKit::toImpl(mainResourceRef), WebKit::toImpl(subresourcesRef), WebKit::toImpl(subframeArchivesRef));
    return WebKit::toAPI(&webArchive.leakRef());
}

WKWebArchiveRef WKWebArchiveCreateWithData(WKDataRef dataRef)
{
    auto webArchive = API::WebArchive::create(WebKit::toImpl(dataRef));
    return WebKit::toAPI(&webArchive.leakRef());
}

WKWebArchiveRef WKWebArchiveCreateFromRange(WKBundleRangeHandleRef rangeHandleRef)
{
    auto webArchive = API::WebArchive::create(makeSimpleRange(WebKit::toImpl(rangeHandleRef)->coreRange()));
    return WebKit::toAPI(&webArchive.leakRef());
}

WKWebArchiveResourceRef WKWebArchiveCopyMainResource(WKWebArchiveRef webArchiveRef)
{
    RefPtr<API::WebArchiveResource> mainResource = WebKit::toImpl(webArchiveRef)->mainResource();
    return WebKit::toAPI(mainResource.leakRef());
}

WKArrayRef WKWebArchiveCopySubresources(WKWebArchiveRef webArchiveRef)
{
    RefPtr<API::Array> subresources = WebKit::toImpl(webArchiveRef)->subresources();
    return WebKit::toAPI(subresources.leakRef());
}

WKArrayRef WKWebArchiveCopySubframeArchives(WKWebArchiveRef webArchiveRef)
{
    RefPtr<API::Array> subframeArchives = WebKit::toImpl(webArchiveRef)->subframeArchives();
    return WebKit::toAPI(subframeArchives.leakRef());
}

WKDataRef WKWebArchiveCopyData(WKWebArchiveRef webArchiveRef)
{
    return WebKit::toAPI(&WebKit::toImpl(webArchiveRef)->data().leakRef());
}
