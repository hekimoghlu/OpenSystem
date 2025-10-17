/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 2, 2023.
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
#pragma once

#include "WKSharedAPICast.h"
#include "WKBundlePage.h"
#include "WKBundlePagePrivate.h"
#include "WKBundlePrivate.h"
#include <WebCore/EditorInsertAction.h>
#include <WebCore/TextAffinity.h>

namespace WebKit {

class InjectedBundle;
class InjectedBundleBackForwardList;
class InjectedBundleBackForwardListItem;
class InjectedBundleCSSStyleDeclarationHandle;
class InjectedBundleDOMWindowExtension;
class InjectedBundleHitTestResult;
class InjectedBundleNodeHandle;
class InjectedBundleRangeHandle;
class InjectedBundleScriptWorld;
class PageBanner;
class WebFrame;
class WebInspector;
class WebPage;
class WebPageGroupProxy;
class WebPageOverlay;

WK_ADD_API_MAPPING(WKBundleBackForwardListItemRef, InjectedBundleBackForwardListItem)
WK_ADD_API_MAPPING(WKBundleBackForwardListRef, InjectedBundleBackForwardList)
WK_ADD_API_MAPPING(WKBundleCSSStyleDeclarationRef, InjectedBundleCSSStyleDeclarationHandle)
WK_ADD_API_MAPPING(WKBundleDOMWindowExtensionRef, InjectedBundleDOMWindowExtension)
WK_ADD_API_MAPPING(WKBundleFrameRef, WebFrame)
WK_ADD_API_MAPPING(WKBundleHitTestResultRef, InjectedBundleHitTestResult)
WK_ADD_API_MAPPING(WKBundleNodeHandleRef, InjectedBundleNodeHandle)
WK_ADD_API_MAPPING(WKBundlePageBannerRef, PageBanner)
WK_ADD_API_MAPPING(WKBundlePageOverlayRef, WebPageOverlay)
WK_ADD_API_MAPPING(WKBundlePageRef, WebPage)
WK_ADD_API_MAPPING(WKBundleRangeHandleRef, InjectedBundleRangeHandle)
WK_ADD_API_MAPPING(WKBundleRef, InjectedBundle)
WK_ADD_API_MAPPING(WKBundleScriptWorldRef, InjectedBundleScriptWorld)

inline WKInsertActionType toAPI(WebCore::EditorInsertAction action)
{
    switch (action) {
    case WebCore::EditorInsertAction::Typed:
        return kWKInsertActionTyped;
    case WebCore::EditorInsertAction::Pasted:
        return kWKInsertActionPasted;
    case WebCore::EditorInsertAction::Dropped:
        return kWKInsertActionDropped;
    }
    ASSERT_NOT_REACHED();
    return kWKInsertActionTyped;
}

inline WKAffinityType toAPI(WebCore::Affinity affinity)
{
    switch (affinity) {
    case WebCore::Affinity::Upstream:
        return kWKAffinityUpstream;
    case WebCore::Affinity::Downstream:
        return kWKAffinityDownstream;
    }
    ASSERT_NOT_REACHED();
    return kWKAffinityUpstream;
}

} // namespace WebKit
