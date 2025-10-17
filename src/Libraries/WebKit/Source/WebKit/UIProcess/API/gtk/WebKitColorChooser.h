/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 11, 2022.
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

#include "WebColorPickerGtk.h"
#include <WebCore/IntRect.h>
#include <wtf/glib/GRefPtr.h>

typedef struct _WebKitColorChooserRequest WebKitColorChooserRequest;

namespace WebCore {
class Color;
}

namespace WebKit {

class WebKitColorChooser final : public WebColorPickerGtk {
public:
    static Ref<WebKitColorChooser> create(WebPageProxy&, const WebCore::Color&, const WebCore::IntRect&);
    virtual ~WebKitColorChooser();

    const WebCore::IntRect& elementRect() const { return m_elementRect; }

private:
    WebKitColorChooser(WebPageProxy&, const WebCore::Color&, const WebCore::IntRect&);

    void endPicker() override;
    void showColorPicker(const WebCore::Color&) override;

    static void colorChooserRequestFinished(WebKitColorChooserRequest*, WebKitColorChooser*);
    static void colorChooserRequestRGBAChanged(WebKitColorChooserRequest*, GParamSpec*, WebKitColorChooser*);

    GRefPtr<WebKitColorChooserRequest> m_request;
    WebCore::IntRect m_elementRect;
};

} // namespace WebKit
