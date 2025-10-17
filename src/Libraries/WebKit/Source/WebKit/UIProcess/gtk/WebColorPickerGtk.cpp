/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 8, 2023.
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
#include "WebColorPickerGtk.h"

#include "WebPageProxy.h"
#include <WebCore/Color.h>
#include <WebCore/GtkUtilities.h>
#include <WebCore/GtkVersioning.h>
#include <glib/gi18n-lib.h>

namespace WebKit {
using namespace WebCore;

Ref<WebColorPickerGtk> WebColorPickerGtk::create(WebPageProxy& page, const Color& initialColor, const IntRect& rect)
{
    return adoptRef(*new WebColorPickerGtk(page, initialColor, rect));
}

WebColorPickerGtk::WebColorPickerGtk(WebPageProxy& page, const Color& initialColor, const IntRect&)
    : WebColorPicker(&page.colorPickerClient())
    , m_initialColor(initialColor)
    , m_webView(page.viewWidget())
    , m_colorChooser(nullptr)
{
}

WebColorPickerGtk::~WebColorPickerGtk()
{
    endPicker();
}

void WebColorPickerGtk::cancel()
{
    setSelectedColor(m_initialColor);
}

void WebColorPickerGtk::endPicker()
{
    if (!m_colorChooser)
        return;

    gtk_widget_destroy(m_colorChooser);
    m_colorChooser = nullptr;
}

void WebColorPickerGtk::didChooseColor(const Color& color)
{
    if (CheckedPtr client = this->client())
        client->didChooseColor(color);
}

void WebColorPickerGtk::colorChooserDialogRGBAChangedCallback(GtkColorChooser* colorChooser, GParamSpec*, WebColorPickerGtk* colorPicker)
{
    GdkRGBA rgba;
    gtk_color_chooser_get_rgba(colorChooser, &rgba);
    colorPicker->didChooseColor(rgba);
}

void WebColorPickerGtk::colorChooserDialogResponseCallback(GtkColorChooser*, int responseID, WebColorPickerGtk* colorPicker)
{
    if (responseID != GTK_RESPONSE_OK)
        colorPicker->cancel();
    colorPicker->endPicker();
}

void WebColorPickerGtk::showColorPicker(const Color& color)
{
    if (!client())
        return;

    m_initialColor = color;

    if (!m_colorChooser) {
        GtkWidget* toplevel = gtk_widget_get_toplevel(m_webView);
        m_colorChooser = gtk_color_chooser_dialog_new(_("Select Color"), WebCore::widgetIsOnscreenToplevelWindow(toplevel) ? GTK_WINDOW(toplevel) : nullptr);
        gtk_color_chooser_set_rgba(GTK_COLOR_CHOOSER(m_colorChooser), &m_initialColor);
        g_signal_connect(m_colorChooser, "notify::rgba", G_CALLBACK(WebColorPickerGtk::colorChooserDialogRGBAChangedCallback), this);
        g_signal_connect(m_colorChooser, "response", G_CALLBACK(WebColorPickerGtk::colorChooserDialogResponseCallback), this);
    } else
        gtk_color_chooser_set_rgba(GTK_COLOR_CHOOSER(m_colorChooser), &m_initialColor);

    gtk_widget_show(m_colorChooser);
}

} // namespace WebKit
