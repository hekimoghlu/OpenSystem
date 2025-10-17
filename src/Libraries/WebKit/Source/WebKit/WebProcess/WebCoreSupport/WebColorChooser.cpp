/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 7, 2023.
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
#include "WebColorChooser.h"

#include "ColorControlSupportsAlpha.h"
#include "WebPage.h"
#include "WebPageProxyMessages.h"
#include "WebProcess.h"
#include <WebCore/ColorChooserClient.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebKit {
using namespace WebCore;

WTF_MAKE_TZONE_ALLOCATED_IMPL(WebColorChooser);

WebColorChooser::WebColorChooser(WebPage* page, ColorChooserClient* client, const Color& initialColor)
    : m_colorChooserClient(client)
    , m_page(page)
{
    m_page->setActiveColorChooser(this);
    auto supportsAlpha = m_colorChooserClient->supportsAlpha() ? ColorControlSupportsAlpha::Yes : ColorControlSupportsAlpha::No;
    WebProcess::singleton().parentProcessConnection()->send(Messages::WebPageProxy::ShowColorPicker(initialColor, m_colorChooserClient->elementRectRelativeToRootView(), supportsAlpha, m_colorChooserClient->suggestedColors()), m_page->identifier());
}

WebColorChooser::~WebColorChooser()
{
    if (!m_page)
        return;

    m_page->setActiveColorChooser(nullptr);
}

void WebColorChooser::didChooseColor(const Color& color)
{
    if (RefPtr colorChooserClient = m_colorChooserClient.get())
        colorChooserClient->didChooseColor(color);
}

void WebColorChooser::didEndChooser()
{
    if (RefPtr colorChooserClient = m_colorChooserClient.get())
        colorChooserClient->didEndChooser();
}

void WebColorChooser::disconnectFromPage()
{
    m_page = 0;
}

void WebColorChooser::reattachColorChooser(const Color& color)
{
    ASSERT(m_page);
    m_page->setActiveColorChooser(this);

    RefPtr colorChooserClient = m_colorChooserClient.get();
    ASSERT(colorChooserClient);
    auto supportsAlpha = colorChooserClient->supportsAlpha() ? ColorControlSupportsAlpha::Yes : ColorControlSupportsAlpha::No;
    WebProcess::singleton().parentProcessConnection()->send(Messages::WebPageProxy::ShowColorPicker(color, colorChooserClient->elementRectRelativeToRootView(), supportsAlpha, colorChooserClient->suggestedColors()), m_page->identifier());
}

void WebColorChooser::setSelectedColor(const Color& color)
{
    if (!m_page)
        return;
    
    if (m_page->activeColorChooser() != this)
        return;

    WebProcess::singleton().parentProcessConnection()->send(Messages::WebPageProxy::SetColorPickerColor(color), m_page->identifier());
}

void WebColorChooser::endChooser()
{
    if (!m_page)
        return;

    WebProcess::singleton().parentProcessConnection()->send(Messages::WebPageProxy::EndColorPicker(), m_page->identifier());
}

} // namespace WebKit
