/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 4, 2024.
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

#include <wtf/URL.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

class SharedBuffer;

class PasteboardWriterData final {
public:
    WEBCORE_EXPORT PasteboardWriterData();
    WEBCORE_EXPORT ~PasteboardWriterData();

    WEBCORE_EXPORT bool isEmpty() const;

    struct PlainText {
        bool canSmartCopyOrDelete;
        String text;
    };

    struct WebContent {
        WebContent();
        WEBCORE_EXPORT ~WebContent();

#if PLATFORM(COCOA)
        String contentOrigin;
        bool canSmartCopyOrDelete;
        RefPtr<SharedBuffer> dataInWebArchiveFormat;
        RefPtr<SharedBuffer> dataInRTFDFormat;
        RefPtr<SharedBuffer> dataInRTFFormat;
        RefPtr<SharedBuffer> dataInAttributedStringFormat;
        String dataInHTMLFormat;
        String dataInStringFormat;
        Vector<std::pair<String, RefPtr<WebCore::SharedBuffer>>> clientTypesAndData;
#endif
    };

    const std::optional<PlainText>& plainText() const { return m_plainText; }
    void setPlainText(PlainText);

    struct URLData {
        URL url;
        String title;
#if PLATFORM(MAC)
        String userVisibleForm;
#elif PLATFORM(GTK)
        String markup;
#endif
    };

    const std::optional<URLData>& urlData() const { return m_url; }
    void setURLData(URLData);

    const std::optional<WebContent>& webContent() const { return m_webContent; }
    void setWebContent(WebContent);

private:
    std::optional<PlainText> m_plainText;
    std::optional<URLData> m_url;
    std::optional<WebContent> m_webContent;
};

}
