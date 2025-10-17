/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 27, 2022.
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
#import "PageBanner.h"

#if PLATFORM(MAC)

#import "WebPage.h"
#import <WebCore/GraphicsLayer.h>
#import <WebCore/LocalFrameView.h>
#import <WebCore/Page.h>

namespace WebKit {
using namespace WebCore;

Ref<PageBanner> PageBanner::create(CALayer *layer, int height, std::unique_ptr<Client>&& client)
{
    return adoptRef(*new PageBanner(layer, height, WTFMove(client)));
}

PageBanner::PageBanner(CALayer *layer, int height, std::unique_ptr<Client>&& client)
    : m_client(WTFMove(client))
    , m_layer(layer)
    , m_height(height)
{
}

void PageBanner::addToPage(Type type, WebPage* webPage)
{
    m_type = type;
    m_webPage = webPage;

    ASSERT(m_type != NotSet);
    ASSERT(m_webPage);

    switch (m_type) {
    case Header:
        m_webPage->corePage()->setHeaderHeight(m_height);
        break;
    case Footer:
        m_webPage->corePage()->setFooterHeight(m_height);
        break;
    case NotSet:
        ASSERT_NOT_REACHED();
    }
}

void PageBanner::didAddParentLayer(GraphicsLayer* parentLayer)
{
    if (!parentLayer)
        return;

    m_layer.get().bounds = CGRectMake(0, 0, parentLayer->size().width(), parentLayer->size().height());
    [parentLayer->platformLayer() addSublayer:m_layer.get()];
}

void PageBanner::detachFromPage()
{
    if (!m_webPage)
        return;

    // m_webPage->corePage() can be null when this is called from WebPage::~WebPage() after
    // the web page has been closed.
    if (m_webPage->corePage()) {
        // We can hide the banner by removing the parent layer that hosts it.
        if (m_type == Header)
            m_webPage->corePage()->setHeaderHeight(0);
        else if (m_type == Footer)
            m_webPage->corePage()->setFooterHeight(0);
    }

    m_type = NotSet;
    m_webPage = 0;
}

void PageBanner::hide()
{
    // We can hide the banner by removing the parent layer that hosts it.
    if (m_type == Header)
        m_webPage->corePage()->setHeaderHeight(0);
    else if (m_type == Footer)
        m_webPage->corePage()->setFooterHeight(0);

    m_isHidden = true;
}

void PageBanner::showIfHidden()
{
    if (!m_isHidden)
        return;
    m_isHidden = false;

    // This will re-create a parent layer in the WebCore layer tree, and we will re-add
    // m_layer as a child of it. 
    addToPage(m_type, RefPtr { m_webPage.get() }.get());
}

void PageBanner::didChangeDeviceScaleFactor(float scaleFactor)
{
    m_layer.get().contentsScale = scaleFactor;
    [m_layer setNeedsDisplay];
}

bool PageBanner::mouseEvent(const WebMouseEvent& mouseEvent)
{
    if (m_isHidden)
        return false;

    RefPtr frameView = m_webPage->localMainFrameView();
    if (!frameView)
        return false;

    IntPoint positionInBannerSpace;

    switch (m_type) {
    case Header: {
        positionInBannerSpace = frameView->rootViewToTotalContents(mouseEvent.position());
        break;
    }
    case Footer: {
        positionInBannerSpace = frameView->rootViewToTotalContents(mouseEvent.position()) - IntSize(0, frameView->totalContentsSize().height() - m_height);
        break;
    }
    case NotSet:
        ASSERT_NOT_REACHED();
    }

    if (!m_mouseDownInBanner && (positionInBannerSpace.y() < 0 || positionInBannerSpace.y() > m_height))
        return false;

    if (mouseEvent.type() == WebEventType::MouseDown)
        m_mouseDownInBanner = true;
    else if (mouseEvent.type() == WebEventType::MouseUp)
        m_mouseDownInBanner = false;

    return m_client->mouseEvent(this, mouseEvent.type(), mouseEvent.button(), positionInBannerSpace);
}

CALayer *PageBanner::layer()
{
    return m_layer.get();
}

} // namespace WebKit

#endif // PLATFORM(MAC)
