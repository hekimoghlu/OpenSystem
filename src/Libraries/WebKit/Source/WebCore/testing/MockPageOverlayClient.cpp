/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 1, 2024.
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
#include "MockPageOverlayClient.h"

#include "Document.h"
#include "GraphicsContext.h"
#include "GraphicsLayer.h"
#include "LocalFrame.h"
#include "Page.h"
#include "PageOverlayController.h"
#include "PlatformMouseEvent.h"
#include <wtf/NeverDestroyed.h>
#include <wtf/text/MakeString.h>
#include <wtf/text/StringBuilder.h>

namespace WebCore {

MockPageOverlayClient& MockPageOverlayClient::singleton()
{
    static NeverDestroyed<MockPageOverlayClient> sharedClient;
    return sharedClient.get();
}

MockPageOverlayClient::MockPageOverlayClient() = default;

Ref<MockPageOverlay> MockPageOverlayClient::installOverlay(Page& page, PageOverlay::OverlayType overlayType)
{
    auto overlay = PageOverlay::create(*this, overlayType);
    page.pageOverlayController().installPageOverlay(overlay, PageOverlay::FadeMode::DoNotFade);

    auto mockOverlay = MockPageOverlay::create(overlay.ptr());
    m_overlays.add(mockOverlay.ptr());

    return mockOverlay;
}

void MockPageOverlayClient::uninstallAllOverlays()
{
    while (!m_overlays.isEmpty()) {
        RefPtr<MockPageOverlay> mockOverlay = m_overlays.takeAny();
        PageOverlayController* overlayController = mockOverlay->overlay()->controller();
        ASSERT(overlayController);
        overlayController->uninstallPageOverlay(*mockOverlay->overlay(), PageOverlay::FadeMode::DoNotFade);
    }
}

String MockPageOverlayClient::layerTreeAsText(Page& page, OptionSet<LayerTreeAsTextOptions> options)
{
    GraphicsLayer* viewOverlayRoot = page.pageOverlayController().viewOverlayRootLayer();
    GraphicsLayer* documentOverlayRoot = page.pageOverlayController().documentOverlayRootLayer();
    
    return makeString("View-relative:\n"_s, (viewOverlayRoot ? viewOverlayRoot->layerTreeAsText(options | LayerTreeAsTextOptions::IncludePageOverlayLayers) : "(no view-relative overlay root)"_s)
        , "\n\nDocument-relative:\n"_s, (documentOverlayRoot ? documentOverlayRoot->layerTreeAsText(options | LayerTreeAsTextOptions::IncludePageOverlayLayers) : "(no document-relative overlay root)"_s));
}

void MockPageOverlayClient::willMoveToPage(PageOverlay&, Page*)
{
}

void MockPageOverlayClient::didMoveToPage(PageOverlay& overlay, Page* page)
{
    if (page)
        overlay.setNeedsDisplay();
}

void MockPageOverlayClient::drawRect(PageOverlay& overlay, GraphicsContext& context, const IntRect&)
{
    GraphicsContextStateSaver stateSaver(context);

    FloatRect insetRect = overlay.bounds();

    if (overlay.overlayType() == PageOverlay::OverlayType::Document) {
        context.setStrokeColor(Color::green);
        insetRect.inflate(-50);
    } else {
        context.setStrokeColor(Color::blue);
        insetRect.inflate(-20);
    }

    context.strokeRect(insetRect, 20);
}

bool MockPageOverlayClient::mouseEvent(PageOverlay& overlay, const PlatformMouseEvent& event)
{
    if (auto* localMainFrame = dynamicDowncast<LocalFrame>(overlay.page()->mainFrame())) {
        localMainFrame->protectedDocument()->addConsoleMessage(MessageSource::Other, MessageLevel::Debug,
            makeString("MockPageOverlayClient::mouseEvent location ("_s, event.position().x(), ", "_s, event.position().y(), ')'));
    }
    return false;
}

void MockPageOverlayClient::didScrollFrame(PageOverlay&, LocalFrame&)
{
}

bool MockPageOverlayClient::copyAccessibilityAttributeStringValueForPoint(PageOverlay&, String /* attribute */, FloatPoint, String&)
{
    return false;
}

bool MockPageOverlayClient::copyAccessibilityAttributeBoolValueForPoint(PageOverlay&, String /* attribute */, FloatPoint, bool&)
{
    return false;
}

Vector<String> MockPageOverlayClient::copyAccessibilityAttributeNames(PageOverlay&, bool /* parameterizedNames */)
{
    return Vector<String>();
}

}
