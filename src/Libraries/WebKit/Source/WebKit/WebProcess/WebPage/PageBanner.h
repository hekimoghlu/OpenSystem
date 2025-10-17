/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 25, 2021.
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

#include "APIObject.h"
#include "WebMouseEvent.h"
#include <wtf/WeakPtr.h>

#if PLATFORM(MAC)
OBJC_CLASS CALayer;
#include <wtf/RetainPtr.h>
#endif

namespace WebCore {
class GraphicsLayer;
}

namespace WebKit {

class WebPage;

class PageBanner : public API::ObjectImpl<API::Object::Type::BundlePageBanner> {
public:
    enum Type {
        NotSet,
        Header,
        Footer
    };

    class Client {
    public:
        virtual ~Client() { }
        virtual bool mouseEvent(PageBanner*, WebEventType, WebMouseEventButton, const WebCore::IntPoint&) = 0;
    };

#if PLATFORM(MAC)
    static Ref<PageBanner> create(CALayer *, int height, std::unique_ptr<Client>&&);
    CALayer *layer();
#endif

    virtual ~PageBanner();

    void addToPage(Type, WebPage*);
    void detachFromPage();

    void hide();
    void showIfHidden();

    bool mouseEvent(const WebMouseEvent&);
    void didChangeDeviceScaleFactor(float scaleFactor);

    void didAddParentLayer(WebCore::GraphicsLayer*);

private:
#if PLATFORM(MAC)
    explicit PageBanner(CALayer *, int height, std::unique_ptr<Client>&&);
#endif

    std::unique_ptr<Client> m_client;

#if PLATFORM(MAC)
    Type m_type = NotSet;
    WeakPtr<WebPage> m_webPage;
    bool m_mouseDownInBanner = false;
    bool m_isHidden = false;

    RetainPtr<CALayer> m_layer;
    int m_height;
#endif
};

} // namespace WebKit
