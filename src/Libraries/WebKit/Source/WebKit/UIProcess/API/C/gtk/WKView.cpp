/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 3, 2022.
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
#include "WKView.h"

#include "WKAPICast.h"
#include "WKViewPrivate.h"
#include "WebKitWebViewBasePrivate.h"

using namespace WebKit;

WKViewRef WKViewCreate(WKPageConfigurationRef configuration)
{
    return toAPI(webkitWebViewBaseCreate(*toImpl(configuration)));
}

WKPageRef WKViewGetPage(WKViewRef viewRef)
{
    return toAPI(webkitWebViewBaseGetPage(toImpl(viewRef)));
}

void WKViewSetFocus(WKViewRef viewRef, bool focused)
{
    webkitWebViewBaseSetFocus(toImpl(viewRef), focused);
}

void WKViewSetEditable(WKViewRef viewRef, bool editable)
{
    webkitWebViewBaseSetEditable(toImpl(viewRef), editable);
}

void WKViewSetEnableBackForwardNavigationGesture(WKViewRef viewRef, bool enabled)
{
    webkitWebViewBaseSetEnableBackForwardNavigationGesture(toImpl(viewRef), enabled);
}

bool WKViewBeginBackSwipeForTesting(WKViewRef viewRef)
{
    return webkitWebViewBaseBeginBackSwipeForTesting(toImpl(viewRef));
}

bool WKViewCompleteBackSwipeForTesting(WKViewRef viewRef)
{
    return webkitWebViewBaseCompleteBackSwipeForTesting(toImpl(viewRef));
}

GVariant* WKViewContentsOfUserInterfaceItem(WKViewRef viewRef, const char* userInterfaceItem)
{
    return webkitWebViewBaseContentsOfUserInterfaceItem(toImpl(viewRef), userInterfaceItem);
}
