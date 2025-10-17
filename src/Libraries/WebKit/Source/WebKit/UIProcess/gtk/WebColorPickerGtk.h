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
#pragma once

#include "WebColorPicker.h"
#include <gdk/gdk.h>

typedef struct _GtkColorChooser GtkColorChooser;

namespace WebCore {
class Color;
class IntRect;
}

namespace WebKit {

class WebColorPickerGtk : public WebColorPicker {
public:
    static Ref<WebColorPickerGtk> create(WebPageProxy&, const WebCore::Color&, const WebCore::IntRect&);
    virtual ~WebColorPickerGtk();

    void endPicker() override;
    void showColorPicker(const WebCore::Color&) override;

    void cancel();

    const GdkRGBA* initialColor() const { return &m_initialColor; }

protected:
    WebColorPickerGtk(WebPageProxy&, const WebCore::Color&, const WebCore::IntRect&);

    void didChooseColor(const WebCore::Color&);

    GdkRGBA m_initialColor;
    GtkWidget* m_webView;

private:
    static void colorChooserDialogRGBAChangedCallback(GtkColorChooser*, GParamSpec*, WebColorPickerGtk*);
    static void colorChooserDialogResponseCallback(GtkColorChooser*, int /*responseID*/, WebColorPickerGtk*);

    GtkWidget* m_colorChooser;
};

} // namespace WebKit
