/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 17, 2022.
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

#if PLATFORM(MAC)

#include "APIObject.h"
#include "ContextMenuContextData.h"
#include "WebHitTestResultData.h"
#include <wtf/WeakPtr.h>

namespace WebKit {
class WebPageProxy;
}

namespace API {

class ContextMenuElementInfoMac final : public ObjectImpl<Object::Type::ContextMenuElementInfoMac> {
public:
    template<typename... Args> static Ref<ContextMenuElementInfoMac> create(Args&&... args)
    {
        return adoptRef(*new ContextMenuElementInfoMac(std::forward<Args>(args)...));
    }

    const WebKit::WebHitTestResultData& hitTestResultData() const { return m_hitTestResultData; }
    WebKit::WebPageProxy* page() { return m_page.get(); }
    const WTF::String& qrCodePayloadString() const { return m_qrCodePayloadString; }
    bool hasEntireImage() const { return m_hasEntireImage; }

private:
    ContextMenuElementInfoMac(const WebKit::ContextMenuContextData& data, WebKit::WebPageProxy& page)
        : m_hitTestResultData(data.webHitTestResultData().value())
        , m_page(page)
        , m_qrCodePayloadString(data.qrCodePayloadString())
        , m_hasEntireImage(data.hasEntireImage()) { }

    WebKit::WebHitTestResultData m_hitTestResultData;
    WeakPtr<WebKit::WebPageProxy> m_page;
    WTF::String m_qrCodePayloadString;
    bool m_hasEntireImage { false };
};

} // namespace API

#endif // PLATFORM(MAC)
