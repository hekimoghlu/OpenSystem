/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 21, 2024.
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
#include "WKBundleRangeHandle.h"
#include "WKBundleRangeHandlePrivate.h"

#include "InjectedBundleNodeHandle.h"
#include "InjectedBundleRangeHandle.h"
#include "WKAPICast.h"
#include "WKBundleAPICast.h"
#include "WebFrame.h"
#include "WebImage.h"
#include <WebCore/IntRect.h>

WKTypeID WKBundleRangeHandleGetTypeID()
{
    return WebKit::toAPI(WebKit::InjectedBundleRangeHandle::APIType);
}

WKBundleRangeHandleRef WKBundleRangeHandleCreate(JSContextRef contextRef, JSObjectRef objectRef)
{
    RefPtr<WebKit::InjectedBundleRangeHandle> rangeHandle = WebKit::InjectedBundleRangeHandle::getOrCreate(contextRef, objectRef);
    return toAPI(rangeHandle.leakRef());
}

WKRect WKBundleRangeHandleGetBoundingRectInWindowCoordinates(WKBundleRangeHandleRef rangeHandleRef)
{
    WebCore::IntRect boundingRect = WebKit::toImpl(rangeHandleRef)->boundingRectInWindowCoordinates();
    return WKRectMake(boundingRect.x(), boundingRect.y(), boundingRect.width(), boundingRect.height());
}

WKImageRef WKBundleRangeHandleCopySnapshotWithOptions(WKBundleRangeHandleRef rangeHandleRef, WKSnapshotOptions options)
{
    RefPtr<WebKit::WebImage> image = WebKit::toImpl(rangeHandleRef)->renderedImage(WebKit::toSnapshotOptions(options));
    return toAPI(image.leakRef());
}

WKBundleFrameRef WKBundleRangeHandleCopyDocumentFrame(WKBundleRangeHandleRef rangeHandleRef)
{
    RefPtr frame = WebKit::toImpl(rangeHandleRef)->document()->documentFrame();
    return toAPI(frame.leakRef());
}
