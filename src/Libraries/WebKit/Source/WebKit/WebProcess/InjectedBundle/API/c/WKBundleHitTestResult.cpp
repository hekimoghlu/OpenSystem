/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 15, 2024.
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
#include "WKBundleHitTestResult.h"

#include "InjectedBundleHitTestResult.h"
#include "InjectedBundleNodeHandle.h"
#include "WKAPICast.h"
#include "WKBundleAPICast.h"
#include "WebFrame.h"
#include "WebImage.h"
#include <WebCore/ScriptExecutionContext.h>

WKTypeID WKBundleHitTestResultGetTypeID()
{
    return WebKit::toAPI(WebKit::InjectedBundleHitTestResult::APIType);
}

WKBundleNodeHandleRef WKBundleHitTestResultCopyNodeHandle(WKBundleHitTestResultRef hitTestResultRef)
{
    RefPtr<WebKit::InjectedBundleNodeHandle> nodeHandle = WebKit::toImpl(hitTestResultRef)->nodeHandle();
    return toAPI(nodeHandle.leakRef());
}

WKBundleNodeHandleRef WKBundleHitTestResultCopyURLElementHandle(WKBundleHitTestResultRef hitTestResultRef)
{
    RefPtr<WebKit::InjectedBundleNodeHandle> urlElementNodeHandle = WebKit::toImpl(hitTestResultRef)->urlElementHandle();
    return toAPI(urlElementNodeHandle.leakRef());
}

WKBundleFrameRef WKBundleHitTestResultGetFrame(WKBundleHitTestResultRef hitTestResultRef)
{
    return toAPI(WebKit::toImpl(hitTestResultRef)->frame().get());
}

WKBundleFrameRef WKBundleHitTestResultGetTargetFrame(WKBundleHitTestResultRef hitTestResultRef)
{
    return toAPI(WebKit::toImpl(hitTestResultRef)->targetFrame().get());
}

WKURLRef WKBundleHitTestResultCopyAbsoluteImageURL(WKBundleHitTestResultRef hitTestResultRef)
{
    return WebKit::toCopiedURLAPI(WebKit::toImpl(hitTestResultRef)->absoluteImageURL());
}

WKURLRef WKBundleHitTestResultCopyAbsolutePDFURL(WKBundleHitTestResultRef hitTestResultRef)
{
    return WebKit::toCopiedURLAPI(WebKit::toImpl(hitTestResultRef)->absolutePDFURL());
}

WKURLRef WKBundleHitTestResultCopyAbsoluteLinkURL(WKBundleHitTestResultRef hitTestResultRef)
{
    return WebKit::toCopiedURLAPI(WebKit::toImpl(hitTestResultRef)->absoluteLinkURL());
}

WKURLRef WKBundleHitTestResultCopyAbsoluteMediaURL(WKBundleHitTestResultRef hitTestResultRef)
{
    return WebKit::toCopiedURLAPI(WebKit::toImpl(hitTestResultRef)->absoluteMediaURL());
}

bool WKBundleHitTestResultMediaIsInFullscreen(WKBundleHitTestResultRef hitTestResultRef)
{
    return WebKit::toImpl(hitTestResultRef)->mediaIsInFullscreen();
}

bool WKBundleHitTestResultMediaHasAudio(WKBundleHitTestResultRef hitTestResultRef)
{
    return WebKit::toImpl(hitTestResultRef)->mediaHasAudio();
}

bool WKBundleHitTestResultIsDownloadableMedia(WKBundleHitTestResultRef hitTestResultRef)
{
    return WebKit::toImpl(hitTestResultRef)->isDownloadableMedia();
}

WKBundleHitTestResultMediaType WKBundleHitTestResultGetMediaType(WKBundleHitTestResultRef hitTestResultRef)
{
    return toAPI(WebKit::toImpl(hitTestResultRef)->mediaType());
}

WKRect WKBundleHitTestResultGetImageRect(WKBundleHitTestResultRef hitTestResultRef)
{
    return WebKit::toAPI(WebKit::toImpl(hitTestResultRef)->imageRect());
}

WKImageRef WKBundleHitTestResultCopyImage(WKBundleHitTestResultRef hitTestResultRef)
{
    RefPtr<WebKit::WebImage> webImage = WebKit::toImpl(hitTestResultRef)->image();
    return toAPI(webImage.leakRef());
}

bool WKBundleHitTestResultGetIsSelected(WKBundleHitTestResultRef hitTestResultRef)
{
    return WebKit::toImpl(hitTestResultRef)->isSelected();
}

WKStringRef WKBundleHitTestResultCopyLinkLabel(WKBundleHitTestResultRef hitTestResultRef)
{
    return WebKit::toCopiedAPI(WebKit::toImpl(hitTestResultRef)->linkLabel());
}

WKStringRef WKBundleHitTestResultCopyLinkTitle(WKBundleHitTestResultRef hitTestResultRef)
{
    return WebKit::toCopiedAPI(WebKit::toImpl(hitTestResultRef)->linkTitle());
}

WKStringRef WKBundleHitTestResultCopyLinkSuggestedFilename(WKBundleHitTestResultRef hitTestResultRef)
{
    return WebKit::toCopiedAPI(WebKit::toImpl(hitTestResultRef)->linkSuggestedFilename());
}
