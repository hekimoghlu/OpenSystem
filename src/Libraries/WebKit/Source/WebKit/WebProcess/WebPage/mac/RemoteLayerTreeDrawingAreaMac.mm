/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 5, 2024.
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
#import "config.h"
#import "RemoteLayerTreeDrawingAreaMac.h"

#if PLATFORM(MAC)

#import "Logging.h"
#import "WebPage.h"
#import "WebPageCreationParameters.h"
#import <WebCore/GraphicsLayer.h>
#import <WebCore/LocalFrameView.h>
#import <WebCore/RenderLayerBacking.h>
#import <wtf/TZoneMallocInlines.h>

namespace WebKit {
using namespace WebCore;

WTF_MAKE_TZONE_ALLOCATED_IMPL(RemoteLayerTreeDrawingAreaMac);

RemoteLayerTreeDrawingAreaMac::RemoteLayerTreeDrawingAreaMac(WebPage& webPage, const WebPageCreationParameters& parameters)
    : RemoteLayerTreeDrawingArea(webPage, parameters)
{
    setColorSpace(parameters.colorSpace);
}

RemoteLayerTreeDrawingAreaMac::~RemoteLayerTreeDrawingAreaMac() = default;

DelegatedScrollingMode RemoteLayerTreeDrawingAreaMac::delegatedScrollingMode() const
{
    return DelegatedScrollingMode::DelegatedToWebKit;
}

void RemoteLayerTreeDrawingAreaMac::setColorSpace(std::optional<WebCore::DestinationColorSpace> colorSpace)
{
    m_displayColorSpace = colorSpace;

    // We rely on the fact that the full style recalc that happens when moving a window between displays triggers repaints,
    // which causes PlatformCALayerRemote::updateBackingStore() to re-create backing stores with the new colorspace.
}

std::optional<WebCore::DestinationColorSpace> RemoteLayerTreeDrawingAreaMac::displayColorSpace() const
{
    return m_displayColorSpace;
}

void RemoteLayerTreeDrawingAreaMac::mainFrameContentSizeChanged(WebCore::FrameIdentifier, const WebCore::IntSize&)
{
    // Do nothing. This is only relevant to DelegatedToNativeScrollView implementations.
}

void RemoteLayerTreeDrawingAreaMac::adjustTransientZoom(double scale, WebCore::FloatPoint origin)
{
    LOG_WITH_STREAM(ViewGestures, stream << "RemoteLayerTreeDrawingAreaMac::adjustTransientZoom - scale " << scale << " origin " << origin);

    auto totalScale = scale * m_webPage->viewScaleFactor();

    // FIXME: Need to trigger some re-rendering here to render at the new scale, so tiles update while zooming.

    prepopulateRectForZoom(totalScale, origin);
}

void RemoteLayerTreeDrawingAreaMac::willCommitLayerTree(RemoteLayerTreeTransaction& transaction)
{
    // FIXME: Probably need something here for PDF.
    RefPtr frameView = m_webPage->localMainFrameView();
    if (!frameView)
        return;

    RefPtr renderViewGraphicsLayer = frameView->graphicsLayerForPageScale();
    if (!renderViewGraphicsLayer)
        return;

    transaction.setPageScalingLayerID(renderViewGraphicsLayer->primaryLayerID());

    RefPtr scrolledContentsLayer = frameView->graphicsLayerForScrolledContents();
    if (!scrolledContentsLayer)
        return;

    transaction.setScrolledContentsLayerID(scrolledContentsLayer->primaryLayerID());
}

} // namespace WebKit

#endif // PLATFORM(MAC)
