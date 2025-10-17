/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 22, 2023.
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
#include "WKInspector.h"

#if !PLATFORM(IOS_FAMILY)

#include "WKAPICast.h"
#include "WebFrameProxy.h"
#include "WebInspectorUIProxy.h"
#include "WebPageProxy.h"

using namespace WebKit;

WKTypeID WKInspectorGetTypeID()
{
    return toAPI(WebInspectorUIProxy::APIType);
}

WKPageRef WKInspectorGetPage(WKInspectorRef inspectorRef)
{
    return toAPI(toImpl(inspectorRef)->protectedInspectedPage().get());
}

bool WKInspectorIsConnected(WKInspectorRef inspectorRef)
{
    return toImpl(inspectorRef)->isConnected();
}

bool WKInspectorIsVisible(WKInspectorRef inspectorRef)
{
    return toImpl(inspectorRef)->isVisible();
}

bool WKInspectorIsFront(WKInspectorRef inspectorRef)
{
    return toImpl(inspectorRef)->isFront();
}

void WKInspectorConnect(WKInspectorRef inspectorRef)
{
    toImpl(inspectorRef)->connect();
}

void WKInspectorShow(WKInspectorRef inspectorRef)
{
    toImpl(inspectorRef)->show();
}

void WKInspectorHide(WKInspectorRef inspectorRef)
{
    toImpl(inspectorRef)->hide();
}

void WKInspectorClose(WKInspectorRef inspectorRef)
{
    toImpl(inspectorRef)->close();
}

void WKInspectorShowConsole(WKInspectorRef inspectorRef)
{
    toImpl(inspectorRef)->showConsole();
}

void WKInspectorShowResources(WKInspectorRef inspectorRef)
{
    toImpl(inspectorRef)->showResources();
}

void WKInspectorShowMainResourceForFrame(WKInspectorRef inspectorRef, WKFrameRef frameRef)
{
    toImpl(inspectorRef)->showMainResourceForFrame(toImpl(frameRef)->frameID());
}

bool WKInspectorIsAttached(WKInspectorRef inspectorRef)
{
    return toImpl(inspectorRef)->isAttached();
}

void WKInspectorAttach(WKInspectorRef inspectorRef)
{
    auto inspector = toImpl(inspectorRef);
    inspector->attach(inspector->attachmentSide());
}

void WKInspectorDetach(WKInspectorRef inspectorRef)
{
    toImpl(inspectorRef)->detach();
}

bool WKInspectorIsProfilingPage(WKInspectorRef inspectorRef)
{
    return toImpl(inspectorRef)->isProfilingPage();
}

void WKInspectorTogglePageProfiling(WKInspectorRef inspectorRef)
{
    toImpl(inspectorRef)->togglePageProfiling();
}

bool WKInspectorIsElementSelectionActive(WKInspectorRef inspectorRef)
{
    return toImpl(inspectorRef)->isElementSelectionActive();
}

void WKInspectorToggleElementSelection(WKInspectorRef inspectorRef)
{
    toImpl(inspectorRef)->toggleElementSelection();
}

#endif // !PLATFORM(IOS_FAMILY)
