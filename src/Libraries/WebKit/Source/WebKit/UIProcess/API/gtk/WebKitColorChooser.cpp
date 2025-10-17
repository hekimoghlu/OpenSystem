/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 4, 2021.
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
#include "WebKitColorChooser.h"

#include "WebKitColorChooserRequestPrivate.h"
#include "WebKitWebViewPrivate.h"
#include <WebCore/Color.h>
#include <WebCore/IntRect.h>

namespace WebKit {
using namespace WebCore;

Ref<WebKitColorChooser> WebKitColorChooser::create(WebPageProxy& page, const WebCore::Color& initialColor, const WebCore::IntRect& rect)
{
    return adoptRef(*new WebKitColorChooser(page, initialColor, rect));
}

WebKitColorChooser::WebKitColorChooser(WebPageProxy& page, const Color& initialColor, const IntRect& rect)
    : WebColorPickerGtk(page, initialColor, rect)
    , m_elementRect(rect)
{
}

WebKitColorChooser::~WebKitColorChooser()
{
    endPicker();
}

void WebKitColorChooser::endPicker()
{
    if (!m_request) {
        WebColorPickerGtk::endPicker();
        return;
    }

    webkit_color_chooser_request_finish(m_request.get());
}

void WebKitColorChooser::colorChooserRequestFinished(WebKitColorChooserRequest*, WebKitColorChooser* colorChooser)
{
    colorChooser->m_request = nullptr;
}

void WebKitColorChooser::colorChooserRequestRGBAChanged(WebKitColorChooserRequest* request, GParamSpec*, WebKitColorChooser* colorChooser)
{
    GdkRGBA rgba;
    webkit_color_chooser_request_get_rgba(request, &rgba);
    colorChooser->didChooseColor(rgba);
}

void WebKitColorChooser::showColorPicker(const Color& color)
{
    m_initialColor = color;
    GRefPtr<WebKitColorChooserRequest> request = adoptGRef(webkitColorChooserRequestCreate(this));
    g_signal_connect(request.get(), "notify::rgba", G_CALLBACK(WebKitColorChooser::colorChooserRequestRGBAChanged), this);
    g_signal_connect(request.get(), "finished", G_CALLBACK(WebKitColorChooser::colorChooserRequestFinished), this);

    if (webkitWebViewEmitRunColorChooser(WEBKIT_WEB_VIEW(m_webView), request.get()))
        m_request = request;
    else
        WebColorPickerGtk::showColorPicker(color);
}

} // namespace WebKit
