/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 9, 2023.
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
#include "WKBundleFrame.h"
#include "WKBundleFramePrivate.h"

#include "APIArray.h"
#include "APISecurityOrigin.h"
#include "InjectedBundleHitTestResult.h"
#include "InjectedBundleNodeHandle.h"
#include "InjectedBundleRangeHandle.h"
#include "InjectedBundleScriptWorld.h"
#include "WKAPICast.h"
#include "WKBundleAPICast.h"
#include "WKData.h"
#include "WebFrame.h"
#include "WebPage.h"
#include <WebCore/AXObjectCache.h>
#include <WebCore/Document.h>
#include <WebCore/DocumentInlines.h>
#include <WebCore/FocusController.h>
#include <WebCore/FrameLoader.h>
#include <WebCore/LocalFrame.h>
#include <WebCore/LocalFrameView.h>
#include <WebCore/Page.h>
#include <WebCore/ReportingScope.h>

WKTypeID WKBundleFrameGetTypeID()
{
    return WebKit::toAPI(WebKit::WebFrame::APIType);
}

bool WKBundleFrameIsMainFrame(WKBundleFrameRef frameRef)
{
    return WebKit::toImpl(frameRef)->isMainFrame();
}

WKBundleFrameRef WKBundleFrameGetParentFrame(WKBundleFrameRef frameRef)
{
    return toAPI(WebKit::toImpl(frameRef)->parentFrame().get());
}

WKURLRef WKBundleFrameCopyURL(WKBundleFrameRef frameRef)
{
    return WebKit::toCopiedURLAPI(WebKit::toImpl(frameRef)->url());
}

WKURLRef WKBundleFrameCopyProvisionalURL(WKBundleFrameRef frameRef)
{
    return WebKit::toCopiedURLAPI(WebKit::toImpl(frameRef)->provisionalURL());
}

WKFrameLoadState WKBundleFrameGetFrameLoadState(WKBundleFrameRef frameRef)
{
    auto* coreFrame = WebKit::toImpl(frameRef)->coreLocalFrame();
    if (!coreFrame)
        return kWKFrameLoadStateFinished;

    switch (coreFrame->loader().state()) {
    case WebCore::FrameState::Provisional:
        return kWKFrameLoadStateProvisional;
    case WebCore::FrameState::CommittedPage:
        return kWKFrameLoadStateCommitted;
    case WebCore::FrameState::Complete:
        return kWKFrameLoadStateFinished;
    }

    ASSERT_NOT_REACHED();
    return kWKFrameLoadStateFinished;
}

WKArrayRef WKBundleFrameCopyChildFrames(WKBundleFrameRef frameRef)
{
    return WebKit::toAPI(&WebKit::toImpl(frameRef)->childFrames().leakRef());    
}

JSGlobalContextRef WKBundleFrameGetJavaScriptContext(WKBundleFrameRef frameRef)
{
    return WebKit::toImpl(frameRef)->jsContext();
}

WKBundleFrameRef WKBundleFrameForJavaScriptContext(JSContextRef context)
{
    return toAPI(WebKit::WebFrame::frameForContext(context).get());
}

JSGlobalContextRef WKBundleFrameGetJavaScriptContextForWorld(WKBundleFrameRef frameRef, WKBundleScriptWorldRef worldRef)
{
    return WebKit::toImpl(frameRef)->jsContextForWorld(WebKit::toImpl(worldRef));
}

JSValueRef WKBundleFrameGetJavaScriptWrapperForNodeForWorld(WKBundleFrameRef frameRef, WKBundleNodeHandleRef nodeHandleRef, WKBundleScriptWorldRef worldRef)
{
    return WebKit::toImpl(frameRef)->jsWrapperForWorld(WebKit::toImpl(nodeHandleRef), WebKit::toImpl(worldRef));
}

JSValueRef WKBundleFrameGetJavaScriptWrapperForRangeForWorld(WKBundleFrameRef frameRef, WKBundleRangeHandleRef rangeHandleRef, WKBundleScriptWorldRef worldRef)
{
    return WebKit::toImpl(frameRef)->jsWrapperForWorld(WebKit::toImpl(rangeHandleRef), WebKit::toImpl(worldRef));
}

WKStringRef WKBundleFrameCopyName(WKBundleFrameRef frameRef)
{
    return WebKit::toCopiedAPI(WebKit::toImpl(frameRef)->name());
}

WKStringRef WKBundleFrameCopyCounterValue(WKBundleFrameRef frameRef, JSObjectRef element)
{
    return WebKit::toCopiedAPI(WebKit::toImpl(frameRef)->counterValue(element));
}

unsigned WKBundleFrameGetPendingUnloadCount(WKBundleFrameRef frameRef)
{
    return WebKit::toImpl(frameRef)->pendingUnloadCount();
}

WKBundlePageRef WKBundleFrameGetPage(WKBundleFrameRef frameRef)
{
    return toAPI(WebKit::toImpl(frameRef)->page());
}

void WKBundleFrameStopLoading(WKBundleFrameRef frameRef)
{
    WebKit::toImpl(frameRef)->stopLoading();
}

WKStringRef WKBundleFrameCopyLayerTreeAsText(WKBundleFrameRef frameRef)
{
    return WebKit::toCopiedAPI(WebKit::toImpl(frameRef)->layerTreeAsText());
}

bool WKBundleFrameAllowsFollowingLink(WKBundleFrameRef frameRef, WKURLRef urlRef)
{
    return WebKit::toImpl(frameRef)->allowsFollowingLink(URL { WebKit::toWTFString(urlRef) });
}

bool WKBundleFrameHandlesPageScaleGesture(WKBundleFrameRef)
{
    // Deprecated, always returns false, but result is not meaningful.
    return false;
}

WKRect WKBundleFrameGetContentBounds(WKBundleFrameRef frameRef)
{
    return WebKit::toAPI(WebKit::toImpl(frameRef)->contentBounds());
}

WKRect WKBundleFrameGetVisibleContentBounds(WKBundleFrameRef frameRef)
{
    return WebKit::toAPI(WebKit::toImpl(frameRef)->visibleContentBounds());
}

WKRect WKBundleFrameGetVisibleContentBoundsExcludingScrollbars(WKBundleFrameRef frameRef)
{
    return WebKit::toAPI(WebKit::toImpl(frameRef)->visibleContentBoundsExcludingScrollbars());
}

WKSize WKBundleFrameGetScrollOffset(WKBundleFrameRef frameRef)
{
    return WebKit::toAPI(WebKit::toImpl(frameRef)->scrollOffset());
}

bool WKBundleFrameHasHorizontalScrollbar(WKBundleFrameRef frameRef)
{
    return WebKit::toImpl(frameRef)->hasHorizontalScrollbar();
}

bool WKBundleFrameHasVerticalScrollbar(WKBundleFrameRef frameRef)
{
    return WebKit::toImpl(frameRef)->hasVerticalScrollbar();
}

bool WKBundleFrameGetDocumentBackgroundColor(WKBundleFrameRef frameRef, double* red, double* green, double* blue, double* alpha)
{
    return WebKit::toImpl(frameRef)->getDocumentBackgroundColor(red, green, blue, alpha);
}

WKStringRef WKBundleFrameCopySuggestedFilenameForResourceWithURL(WKBundleFrameRef frameRef, WKURLRef urlRef)
{
    return WebKit::toCopiedAPI(WebKit::toImpl(frameRef)->suggestedFilenameForResourceWithURL(URL { WebKit::toWTFString(urlRef) }));
}

WKStringRef WKBundleFrameCopyMIMETypeForResourceWithURL(WKBundleFrameRef frameRef, WKURLRef urlRef)
{
    return WebKit::toCopiedAPI(WebKit::toImpl(frameRef)->mimeTypeForResourceWithURL(URL { WebKit::toWTFString(urlRef) }));
}

bool WKBundleFrameContainsAnyFormElements(WKBundleFrameRef frameRef)
{
    return WebKit::toImpl(frameRef)->containsAnyFormElements();
}

bool WKBundleFrameContainsAnyFormControls(WKBundleFrameRef frameRef)
{
    return WebKit::toImpl(frameRef)->containsAnyFormControls();
}

void WKBundleFrameSetTextDirection(WKBundleFrameRef frameRef, WKStringRef directionRef)
{
    if (!frameRef)
        return;

    WebKit::toImpl(frameRef)->setTextDirection(WebKit::toWTFString(directionRef));
}

void WKBundleFrameSetAccessibleName(WKBundleFrameRef frameRef, WKStringRef accessibleNameRef)
{
    if (!frameRef)
        return;

    WebKit::toImpl(frameRef)->setAccessibleName(AtomString { WebKit::toWTFString(accessibleNameRef) });
}

WKDataRef WKBundleFrameCopyWebArchive(WKBundleFrameRef frameRef)
{
    return WKBundleFrameCopyWebArchiveFilteringSubframes(frameRef, 0, 0);
}

WKDataRef WKBundleFrameCopyWebArchiveFilteringSubframes(WKBundleFrameRef frameRef, WKBundleFrameFrameFilterCallback frameFilterCallback, void* context)
{
#if PLATFORM(COCOA)
    RetainPtr<CFDataRef> data = WebKit::toImpl(frameRef)->webArchiveData(frameFilterCallback, context);
    if (data)
        return WKDataCreate(CFDataGetBytePtr(data.get()), CFDataGetLength(data.get()));
#else
    UNUSED_PARAM(frameRef);
    UNUSED_PARAM(frameFilterCallback);
    UNUSED_PARAM(context);
#endif
    
    return 0;
}

bool WKBundleFrameCallShouldCloseOnWebView(WKBundleFrameRef frameRef)
{
    if (!frameRef)
        return true;

    auto* coreFrame = WebKit::toImpl(frameRef)->coreLocalFrame();
    if (!coreFrame)
        return true;

    return coreFrame->loader().shouldClose();
}

WKBundleHitTestResultRef WKBundleFrameCreateHitTestResult(WKBundleFrameRef frameRef, WKPoint point)
{
    ASSERT(frameRef);
    return WebKit::toAPI(WebKit::toImpl(frameRef)->hitTest(WebKit::toIntPoint(point)).leakRef());
}

WKSecurityOriginRef WKBundleFrameCopySecurityOrigin(WKBundleFrameRef frameRef)
{
    auto* coreFrame = WebKit::toImpl(frameRef)->coreLocalFrame();
    if (!coreFrame)
        return 0;

    return WebKit::toCopiedAPI(&coreFrame->document()->securityOrigin());
}

void WKBundleFrameFocus(WKBundleFrameRef frameRef)
{
    RefPtr coreFrame = WebKit::toImpl(frameRef)->coreLocalFrame();
    if (!coreFrame)
        return;

    coreFrame->page()->checkedFocusController()->setFocusedFrame(coreFrame.get());
}

void _WKBundleFrameGenerateTestReport(WKBundleFrameRef frameRef, WKStringRef message, WKStringRef group)
{
    if (!frameRef)
        return;

    RefPtr coreFrame = WebKit::toImpl(frameRef)->coreLocalFrame();
    if (!coreFrame)
        return;

    if (RefPtr document = coreFrame->document())
        document->reportingScope().generateTestReport(WebKit::toWTFString(message), WebKit::toWTFString(group));
}

void* WKAccessibilityRootObject(WKBundleFrameRef frameRef)
{
    if (!frameRef)
        return nullptr;

    RefPtr frame = WebKit::toImpl(frameRef)->coreLocalFrame();
    if (!frame)
        return nullptr;

    WebCore::AXObjectCache::enableAccessibility();

    RefPtr document = frame->rootFrame().document();
    if (!document)
        return nullptr;

    CheckedPtr axObjectCache = document->axObjectCache();
    if (!axObjectCache)
        return nullptr;

    auto* root = axObjectCache->rootObject();
    if (!root)
        return nullptr;

    return root->wrapper();
}
