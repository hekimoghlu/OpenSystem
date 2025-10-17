/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 20, 2022.
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
#include "WKHitTestResult.h"

#include "APIHitTestResult.h"
#include "WKAPICast.h"

using namespace WebKit;

WKTypeID WKHitTestResultGetTypeID()
{
    return toAPI(API::HitTestResult::APIType);
}

WKURLRef WKHitTestResultCopyAbsoluteImageURL(WKHitTestResultRef hitTestResultRef)
{
    return toCopiedURLAPI(toImpl(hitTestResultRef)->absoluteImageURL());
}

WKURLRef WKHitTestResultCopyAbsolutePDFURL(WKHitTestResultRef hitTestResultRef)
{
    return toCopiedURLAPI(toImpl(hitTestResultRef)->absolutePDFURL());
}

WKURLRef WKHitTestResultCopyAbsoluteLinkURL(WKHitTestResultRef hitTestResultRef)
{
    return toCopiedURLAPI(toImpl(hitTestResultRef)->absoluteLinkURL());
}

WKURLRef WKHitTestResultCopyAbsoluteMediaURL(WKHitTestResultRef hitTestResultRef)
{
    return toCopiedURLAPI(toImpl(hitTestResultRef)->absoluteMediaURL());
}

WKStringRef WKHitTestResultCopyLinkLabel(WKHitTestResultRef hitTestResultRef)
{
    return toCopiedAPI(toImpl(hitTestResultRef)->linkLabel());
}

WKStringRef WKHitTestResultCopyLinkTitle(WKHitTestResultRef hitTestResultRef)
{
    return toCopiedAPI(toImpl(hitTestResultRef)->linkTitle());
}

WKStringRef WKHitTestResultCopyLookupText(WKHitTestResultRef hitTestResult)
{
    return toCopiedAPI(toImpl(hitTestResult)->lookupText());
}

bool WKHitTestResultIsContentEditable(WKHitTestResultRef hitTestResultRef)
{
    return toImpl(hitTestResultRef)->isContentEditable();
}

WKRect WKHitTestResultGetElementBoundingBox(WKHitTestResultRef hitTestResultRef)
{
    return toAPI(toImpl(hitTestResultRef)->elementBoundingBox());
}
