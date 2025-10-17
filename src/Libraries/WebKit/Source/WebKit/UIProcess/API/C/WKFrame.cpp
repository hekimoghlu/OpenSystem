/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 11, 2022.
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
#include "WKFrame.h"

#include "APIData.h"
#include "APIFrameHandle.h"
#include "APIFrameInfo.h"
#include "WKAPICast.h"
#include "WebFrameProxy.h"
#include "WebPageProxy.h"

using namespace WebKit;

WKTypeID WKFrameGetTypeID()
{
    return toAPI(WebFrameProxy::APIType);
}

bool WKFrameIsMainFrame(WKFrameRef frameRef)
{
    return toImpl(frameRef)->isMainFrame();
}

WKFrameLoadState WKFrameGetFrameLoadState(WKFrameRef frameRef)
{
    WebFrameProxy* frame = toImpl(frameRef);
    switch (frame->frameLoadState().state()) {
    case FrameLoadState::State::Provisional:
        return kWKFrameLoadStateProvisional;
    case FrameLoadState::State::Committed:
        return kWKFrameLoadStateCommitted;
    case FrameLoadState::State::Finished:
        return kWKFrameLoadStateFinished;
    }
    
    ASSERT_NOT_REACHED();
    return kWKFrameLoadStateFinished;
}

WKURLRef WKFrameCopyProvisionalURL(WKFrameRef frameRef)
{
    return toCopiedURLAPI(toImpl(frameRef)->provisionalURL());
}

WKURLRef WKFrameCopyURL(WKFrameRef frameRef)
{
    return toCopiedURLAPI(toImpl(frameRef)->url());
}

WKURLRef WKFrameCopyUnreachableURL(WKFrameRef frameRef)
{
    return toCopiedURLAPI(toImpl(frameRef)->unreachableURL());
}

void WKFrameStopLoading(WKFrameRef)
{
}

WKStringRef WKFrameCopyMIMEType(WKFrameRef frameRef)
{
    return toCopiedAPI(toImpl(frameRef)->mimeType());
}

WKStringRef WKFrameCopyTitle(WKFrameRef frameRef)
{
    return toCopiedAPI(toImpl(frameRef)->title());
}

WKPageRef WKFrameGetPage(WKFrameRef frameRef)
{
    return toAPI(toImpl(frameRef)->page());
}

WKCertificateInfoRef WKFrameGetCertificateInfo(WKFrameRef frameRef)
{
    return nullptr;
}

bool WKFrameCanProvideSource(WKFrameRef frameRef)
{
    return toImpl(frameRef)->canProvideSource();
}

bool WKFrameCanShowMIMEType(WKFrameRef, WKStringRef)
{
    return false;
}

bool WKFrameIsDisplayingStandaloneImageDocument(WKFrameRef frameRef)
{
    return toImpl(frameRef)->isDisplayingStandaloneImageDocument();
}

bool WKFrameIsDisplayingMarkupDocument(WKFrameRef frameRef)
{
    return toImpl(frameRef)->isDisplayingMarkupDocument();
}

bool WKFrameIsFrameSet(WKFrameRef frameRef)
{
    return false;
}

WKFrameHandleRef WKFrameCreateFrameHandle(WKFrameRef frameRef)
{
    return toAPI(&API::FrameHandle::create(toImpl(frameRef)->frameID()).leakRef());
}

WKFrameInfoRef WKFrameCreateFrameInfo(WKFrameRef frameRef)
{
    return nullptr;
}

void WKFrameGetMainResourceData(WKFrameRef frameRef, WKFrameGetResourceDataFunction callback, void* context)
{
    toImpl(frameRef)->getMainResourceData([context, callback] (API::Data* data) {
        callback(toAPI(data), nullptr, context);
    });
}

void WKFrameGetResourceData(WKFrameRef frameRef, WKURLRef resourceURL, WKFrameGetResourceDataFunction callback, void* context)
{
    toImpl(frameRef)->getResourceData(toImpl(resourceURL), [context, callback] (API::Data* data) {
        callback(toAPI(data), nullptr, context);
    });
}

void WKFrameGetWebArchive(WKFrameRef frameRef, WKFrameGetWebArchiveFunction callback, void* context)
{
    toImpl(frameRef)->getWebArchive([context, callback] (API::Data* data) {
        callback(toAPI(data), nullptr, context);
    });
}
