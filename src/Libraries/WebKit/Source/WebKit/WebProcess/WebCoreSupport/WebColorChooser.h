/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 6, 2024.
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

#include <WebCore/ColorChooser.h>
#include <wtf/RefCountedAndCanMakeWeakPtr.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/WeakPtr.h>

namespace WebCore {
class Color;
class ColorChooserClient;
}

namespace WebKit {

class WebPage;

class WebColorChooser : public WebCore::ColorChooser, public RefCountedAndCanMakeWeakPtr<WebColorChooser> {
    WTF_MAKE_TZONE_ALLOCATED(WebColorChooser);
public:
    static Ref<WebColorChooser> create(WebPage* page, WebCore::ColorChooserClient* client, const WebCore::Color& initialColor)
    {
        return adoptRef(*new WebColorChooser(page, client, initialColor));
    }

    virtual ~WebColorChooser();

    void ref() const final { RefCounted::ref(); }
    void deref() const final { RefCounted::deref(); }

    void didChooseColor(const WebCore::Color&);
    void didEndChooser();
    void disconnectFromPage();

    void reattachColorChooser(const WebCore::Color&) override;
    void setSelectedColor(const WebCore::Color&) override;
    void endChooser() override;

private:
    WebColorChooser(WebPage*, WebCore::ColorChooserClient*, const WebCore::Color&);

    WeakPtr<WebCore::ColorChooserClient> m_colorChooserClient;
    WeakPtr<WebPage> m_page;
};

} // namespace WebKit
