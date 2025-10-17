/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 1, 2024.
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
#include "PDFDocument.h"

#if ENABLE(PDFJS)

#include "AddEventListenerOptions.h"
#include "DocumentLoader.h"
#include "EventListener.h"
#include "EventNames.h"
#include "FrameDestructionObserverInlines.h"
#include "HTMLAnchorElement.h"
#include "HTMLBodyElement.h"
#include "HTMLHeadElement.h"
#include "HTMLHtmlElement.h"
#include "HTMLIFrameElement.h"
#include "HTMLLinkElement.h"
#include "HTMLNames.h"
#include "HTMLScriptElement.h"
#include "LocalDOMWindow.h"
#include "LocalFrame.h"
#include "RawDataDocumentParser.h"
#include "ScriptController.h"
#include "Settings.h"
#include <JavaScriptCore/ObjectConstructor.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(PDFDocument);

using namespace HTMLNames;

/* PDFDocumentParser: this receives the PDF bytes */

class PDFDocumentParser final : public RawDataDocumentParser {
public:
    static Ref<PDFDocumentParser> create(PDFDocument& document)
    {
        return adoptRef(*new PDFDocumentParser(document));
    }

private:
    explicit PDFDocumentParser(PDFDocument& document)
        : RawDataDocumentParser(document)
    {
    }

    PDFDocument& document() const;

    void appendBytes(DocumentWriter&, std::span<const uint8_t>) override;
    void finish() override;
};

inline PDFDocument& PDFDocumentParser::document() const
{
    // Only used during parsing, so document is guaranteed to be non-null.
    ASSERT(RawDataDocumentParser::document());
    return downcast<PDFDocument>(*RawDataDocumentParser::document());
}

void PDFDocumentParser::appendBytes(DocumentWriter&, std::span<const uint8_t>)
{
    document().updateDuringParsing();
}

void PDFDocumentParser::finish()
{
    document().finishedParsing();
}

/* PDFDocumentEventListener: event listener for the PDFDocument iframe */

class PDFDocumentEventListener final : public EventListener {
public:
    static Ref<PDFDocumentEventListener> create(PDFDocument& document) { return adoptRef(*new PDFDocumentEventListener(document)); }

private:
    explicit PDFDocumentEventListener(PDFDocument& document)
        : EventListener(PDFDocumentEventListenerType)
        , m_document(document)
    {
    }

    bool operator==(const EventListener&) const override;
    void handleEvent(ScriptExecutionContext&, Event&) override;

    WeakPtr<PDFDocument, WeakPtrImplWithEventTargetData> m_document;
};

void PDFDocumentEventListener::handleEvent(ScriptExecutionContext&, Event& event)
{
    if (is<HTMLIFrameElement>(event.target()) && event.type() == eventNames().loadEvent) {
        m_document->injectStyleAndContentScript();
    } else if (is<HTMLScriptElement>(event.target()) && event.type() == eventNames().loadEvent) {
        m_document->setContentScriptLoaded(true);
        if (m_document->isFinishedParsing())
            m_document->finishLoadingPDF();
    } else
        ASSERT_NOT_REACHED();
}

bool PDFDocumentEventListener::operator==(const EventListener& other) const
{
    // All PDFDocumentEventListenerType objects compare as equal; OK since there is only one per document.
    return other.type() == PDFDocumentEventListenerType;
}

/* PDFDocument */

PDFDocument::PDFDocument(LocalFrame& frame, const URL& url)
    : HTMLDocument(&frame, frame.settings(), url, { }, { DocumentClass::PDF })
{
}

PDFDocument::~PDFDocument() = default;

Ref<DocumentParser> PDFDocument::createParser()
{
    return PDFDocumentParser::create(*this);
}

void PDFDocument::createDocumentStructure()
{
    // Description of parameters:
    // - Empty `?file=` parameter prevents default pdf from loading.
    auto viewerURL = "webkit-pdfjs-viewer://pdfjs/web/viewer.html?file="_s;
    Ref rootElement = HTMLHtmlElement::create(*this);
    appendChild(rootElement);

    frame()->injectUserScripts(UserScriptInjectionTime::DocumentStart);

    Ref body = HTMLBodyElement::create(*this);
    body->setAttribute(styleAttr, "margin: 0px;height: 100vh;"_s);
    rootElement->appendChild(body);

    m_iframe = HTMLIFrameElement::create(HTMLNames::iframeTag, *this);
    m_iframe->setAttribute(srcAttr, AtomString(viewerURL));
    m_iframe->setAttribute(styleAttr, "width: 100%; height: 100%; border: 0; display: block;"_s);

    m_listener = PDFDocumentEventListener::create(*this);
    m_iframe->addEventListener(eventNames().loadEvent, *m_listener, false);

    body->appendChild(*m_iframe);
}

void PDFDocument::updateDuringParsing()
{
    if (!m_iframe)
        createDocumentStructure();
}

void PDFDocument::finishedParsing()
{
    ASSERT(m_iframe);
    m_isFinishedParsing = true;
    if (m_isContentScriptLoaded)
        finishLoadingPDF();
}

void PDFDocument::postMessageToIframe(const String& name, JSC::JSObject* data)
{
    auto globalObject = this->globalObject();
    auto& vm = globalObject->vm();
    JSC::JSLockHolder lock(vm);

    JSC::JSObject* message = constructEmptyObject(globalObject);
    message->putDirect(vm, vm.propertyNames->message, JSC::jsNontrivialString(vm, name));
    if (data)
        message->putDirect(vm, JSC::Identifier::fromString(vm, "data"_s), data);

    auto* contentFrame = dynamicDowncast<LocalFrame>(m_iframe->contentFrame());
    if (!contentFrame)
        return;
    auto* contentWindow = contentFrame->window();
    auto* contentWindowGlobalObject = m_iframe->contentDocument()->globalObject();

    WindowPostMessageOptions options;
    if (data)
        options = WindowPostMessageOptions { "/"_s, Vector { JSC::Strong<JSC::JSObject> { vm, data } } };
    auto returnValue = contentWindow->postMessage(*contentWindowGlobalObject, *contentWindow, message, WTFMove(options));
    if (returnValue.hasException())
        returnValue.releaseException();
}

void PDFDocument::sendPDFArrayBuffer()
{
    auto* documentLoader = loader();
    ASSERT(documentLoader);
    if (auto mainResourceData = documentLoader->mainResourceData()) {
        if (auto arrayBuffer = mainResourceData->tryCreateArrayBuffer()) {
            auto& vm = globalObject()->vm();
            JSC::JSLockHolder lock(vm);
            auto* dataObject = JSC::JSArrayBuffer::create(vm, globalObject()->arrayBufferStructure(arrayBuffer->sharingMode()), WTFMove(arrayBuffer));
            postMessageToIframe("open-pdf"_s, dataObject);
        }
    }
}

void PDFDocument::finishLoadingPDF()
{
    sendPDFArrayBuffer();

    if (m_script) {
        m_script->removeEventListener(eventNames().loadEvent, *m_listener, { });
        m_script = nullptr;
    }

    m_listener = nullptr;
}

void PDFDocument::injectStyleAndContentScript()
{
    if (m_injectedStyleAndScript)
        return;

    auto* contentDocument = m_iframe->contentDocument();
    ASSERT(contentDocument->head());
    Ref link = HTMLLinkElement::create(HTMLNames::linkTag, *contentDocument, false);
    link->setAttribute(relAttr, "stylesheet"_s);
#if PLATFORM(COCOA)
    link->setAttribute(hrefAttr, "webkit-pdfjs-viewer://pdfjs/extras/cocoa/style.css"_s);
#elif PLATFORM(GTK) || PLATFORM(WPE)
    link->setAttribute(hrefAttr, "webkit-pdfjs-viewer://pdfjs/extras/adwaita/style.css"_s);
#endif
    contentDocument->head()->appendChild(link);

    ASSERT(contentDocument->body());
    m_script = HTMLScriptElement::create(scriptTag, *contentDocument, false);
    ASSERT(m_listener);
    m_script->addEventListener(eventNames().loadEvent, *m_listener, false);
    m_script->setAttribute(srcAttr, "webkit-pdfjs-viewer://pdfjs/extras/content-script.js"_s);
    contentDocument->body()->appendChild(*m_script);

    m_injectedStyleAndScript = true;
}

} // namepsace WebCore

#endif // ENABLE(PDFJS)
