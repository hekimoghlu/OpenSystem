/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 10, 2025.
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
#include "WKBundleBackForwardListItem.h"

#include "APIArray.h"
#include "WKBundleAPICast.h"

WKTypeID WKBundleBackForwardListItemGetTypeID()
{
    return WebKit::toAPI(API::Object::Type::Null);
}

bool WKBundleBackForwardListItemIsSame(WKBundleBackForwardListItemRef itemRef1, WKBundleBackForwardListItemRef itemRef2)
{
    return false;
}

WKURLRef WKBundleBackForwardListItemCopyOriginalURL(WKBundleBackForwardListItemRef itemRef)
{
    return nullptr;
}

WKURLRef WKBundleBackForwardListItemCopyURL(WKBundleBackForwardListItemRef itemRef)
{
    return nullptr;
}

WKStringRef WKBundleBackForwardListItemCopyTitle(WKBundleBackForwardListItemRef itemRef)
{
    return nullptr;
}

WKStringRef WKBundleBackForwardListItemCopyTarget(WKBundleBackForwardListItemRef itemRef)
{
    return nullptr;
}

bool WKBundleBackForwardListItemIsTargetItem(WKBundleBackForwardListItemRef itemRef)
{
    return false;
}

bool WKBundleBackForwardListItemIsInBackForwardCache(WKBundleBackForwardListItemRef itemRef)
{
    return false;
}

bool WKBundleBackForwardListItemHasCachedPageExpired(WKBundleBackForwardListItemRef itemRef)
{
    return false;
}

WKArrayRef WKBundleBackForwardListItemCopyChildren(WKBundleBackForwardListItemRef itemRef)
{
    return nullptr;
}

