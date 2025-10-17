/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 2, 2024.
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
#include "WKBackForwardListRef.h"

#include "APIArray.h"
#include "WKAPICast.h"
#include "WebBackForwardList.h"
#include "WebFrameProxy.h"

using namespace WebKit;

WKTypeID WKBackForwardListGetTypeID()
{
    return toAPI(WebBackForwardList::APIType);
}

WKBackForwardListItemRef WKBackForwardListGetCurrentItem(WKBackForwardListRef listRef)
{
    return toAPI(toImpl(listRef)->currentItem());
}

WKBackForwardListItemRef WKBackForwardListGetBackItem(WKBackForwardListRef listRef)
{
    return toAPI(toImpl(listRef)->backItem());
}

WKBackForwardListItemRef WKBackForwardListGetForwardItem(WKBackForwardListRef listRef)
{
    return toAPI(toImpl(listRef)->forwardItem());
}

WKBackForwardListItemRef WKBackForwardListGetItemAtIndex(WKBackForwardListRef listRef, int index)
{
    return toAPI(toImpl(listRef)->itemAtIndex(index));
}

void WKBackForwardListClear(WKBackForwardListRef listRef)
{
    toImpl(listRef)->clear();
}

unsigned WKBackForwardListGetBackListCount(WKBackForwardListRef listRef)
{
    return toImpl(listRef)->backListCount();
}

unsigned WKBackForwardListGetForwardListCount(WKBackForwardListRef listRef)
{
    return toImpl(listRef)->forwardListCount();
}

WKArrayRef WKBackForwardListCopyBackListWithLimit(WKBackForwardListRef listRef, unsigned limit)
{
    return toAPI(&toImpl(listRef)->backListAsAPIArrayWithLimit(limit).leakRef());
}

WKArrayRef WKBackForwardListCopyForwardListWithLimit(WKBackForwardListRef listRef, unsigned limit)
{
    return toAPI(&toImpl(listRef)->forwardListAsAPIArrayWithLimit(limit).leakRef());
}
