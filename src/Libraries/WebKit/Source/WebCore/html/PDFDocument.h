/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 5, 2023.
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

#if ENABLE(PDFJS)

#include "HTMLDocument.h"
#include "HTMLScriptElement.h"

namespace WebCore {

class HTMLIFrameElement;
class PDFDocumentEventListener;

class PDFDocument final : public HTMLDocument {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(PDFDocument);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(PDFDocument);
public:
    static Ref<PDFDocument> create(LocalFrame& frame, const URL& url)
    {
        auto document = adoptRef(*new PDFDocument(frame, url));
        document->addToContextsMap();
        return document;
    }

    ~PDFDocument();

    void updateDuringParsing();
    void finishedParsing();
    void injectStyleAndContentScript();

    void postMessageToIframe(const String& name, JSC::JSObject* data);
    void finishLoadingPDF();

    bool isFinishedParsing() const { return m_isFinishedParsing; }
    void setContentScriptLoaded(bool loaded) { m_isContentScriptLoaded = loaded; }

private:
    PDFDocument(LocalFrame&, const URL&);

    Ref<DocumentParser> createParser() override;

    void createDocumentStructure();
    void sendPDFArrayBuffer();
    bool m_injectedStyleAndScript { false };
    bool m_isFinishedParsing { false };
    bool m_isContentScriptLoaded { false };
    RefPtr<HTMLIFrameElement> m_iframe;
    RefPtr<HTMLScriptElement> m_script;
    RefPtr<PDFDocumentEventListener> m_listener;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_BEGIN(WebCore::PDFDocument)
    static bool isType(const WebCore::Document& document) { return document.isPDFDocument(); }
    static bool isType(const WebCore::Node& node)
    {
        auto* document = dynamicDowncast<WebCore::Document>(node);
        return document && isType(*document);
    }
SPECIALIZE_TYPE_TRAITS_END()

#endif // ENABLE(PDFJS)
