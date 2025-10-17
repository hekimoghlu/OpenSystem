/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 28, 2021.
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
#include "WKBackForwardListItemRef.h"

#include "WKAPICast.h"
#include "WebBackForwardListItem.h"

using namespace WebKit;

WKTypeID WKBackForwardListItemGetTypeID()
{
    return WebKit::toAPI(WebBackForwardListItem::APIType);
}

WKURLRef WKBackForwardListItemCopyURL(WKBackForwardListItemRef itemRef)
{
    Ref item = *toImpl(itemRef);
    return WebKit::toCopiedURLAPI(item->url());
}

WKStringRef WKBackForwardListItemCopyTitle(WKBackForwardListItemRef itemRef)
{
    Ref item = *toImpl(itemRef);
    return WebKit::toCopiedAPI(item->title());
}

WKURLRef WKBackForwardListItemCopyOriginalURL(WKBackForwardListItemRef itemRef)
{
    Ref item = *toImpl(itemRef);
    return WebKit::toCopiedURLAPI(item->originalURL());
}
