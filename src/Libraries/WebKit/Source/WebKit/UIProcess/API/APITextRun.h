/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 14, 2025.
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

#include "WebPageProxy.h"
#include <WebCore/FloatRect.h>
#include <wtf/WeakPtr.h>
#include <wtf/text/WTFString.h>

namespace API {

class TextRun final : public ObjectImpl<Object::Type::TextRun> {
public:
    static Ref<TextRun> create(WebKit::WebPageProxy& page, WTF::String&& string, WebCore::FloatRect&& rect)
    {
        return adoptRef(*new TextRun(page, WTFMove(string), WTFMove(rect)));
    }

    const WTF::String& string() const { return m_string; }
    WebCore::FloatRect rectInWebView() const;

private:
    explicit TextRun(WebKit::WebPageProxy& page, WTF::String&& string, WebCore::FloatRect&& rect)
        : m_page { page }
        , m_string { WTFMove(string) }
        , m_rectInRootView { WTFMove(rect) }
    {
    }

    WeakPtr<WebKit::WebPageProxy> m_page;
    WTF::String m_string;
    WebCore::FloatRect m_rectInRootView;
};

} // namespace API
