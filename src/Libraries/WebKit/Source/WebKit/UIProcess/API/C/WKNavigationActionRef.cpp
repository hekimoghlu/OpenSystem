/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 11, 2025.
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
#include "WKNavigationActionRef.h"

#include "APINavigationAction.h"
#include "WKAPICast.h"

WKTypeID WKNavigationActionGetTypeID()
{
    return WebKit::toAPI(API::NavigationAction::APIType);
}

bool WKNavigationActionShouldPerformDownload(WKNavigationActionRef action)
{
    return WebKit::toImpl(action)->shouldPerformDownload();
}

WKURLRequestRef WKNavigationActionCopyRequest(WKNavigationActionRef action)
{
    return WebKit::toAPI(&API::URLRequest::create(WebKit::toImpl(action)->request()).leakRef());
}

bool WKNavigationActionGetShouldOpenExternalSchemes(WKNavigationActionRef action)
{
    return WebKit::toImpl(action)->shouldOpenExternalSchemes();
}

WKFrameInfoRef WKNavigationActionCopyTargetFrameInfo(WKNavigationActionRef action)
{
    RefPtr targetFrame = WebKit::toImpl(action)->targetFrame();
    return targetFrame ? WebKit::toAPI(targetFrame.leakRef()) : nullptr;
}

WKFrameNavigationType WKNavigationActionGetNavigationType(WKNavigationActionRef action)
{
    return WebKit::toAPI(WebKit::toImpl(action)->navigationType());
}

WK_EXPORT bool WKNavigationActionHasUnconsumedUserGesture(WKNavigationActionRef action)
{
    return WebKit::toImpl(action)->isProcessingUnconsumedUserGesture();
}
