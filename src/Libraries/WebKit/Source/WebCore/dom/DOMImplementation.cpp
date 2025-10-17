/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 14, 2024.
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
#include "DOMImplementation.h"

#include "CSSStyleSheet.h"
#include "ContentType.h"
#include "DeprecatedGlobalSettings.h"
#include "DocumentType.h"
#include "Element.h"
#include "FTPDirectoryDocument.h"
#include "FrameLoader.h"
#include "HTMLDocument.h"
#include "HTMLHeadElement.h"
#include "HTMLTitleElement.h"
#include "Image.h"
#include "ImageDocument.h"
#include "LocalFrame.h"
#include "LocalFrameLoaderClient.h"
#include "MIMETypeRegistry.h"
#include "MediaDocument.h"
#include "MediaPlayer.h"
#include "MediaQueryParser.h"
#include "PDFDocument.h"
#include "Page.h"
#include "ParserContentPolicy.h"
#include "PluginData.h"
#include "PluginDocument.h"
#include "SVGDocument.h"
#include "SVGNames.h"
#include "SecurityOrigin.h"
#include "SecurityOriginPolicy.h"
#include "Settings.h"
#include "StyleSheetContents.h"
#include "Text.h"
#include "TextDocument.h"
#include "XMLDocument.h"
#include <wtf/StdLibExtras.h>
#include <wtf/TZoneMallocInlines.h>

#if ENABLE(MODEL_ELEMENT)
#include "ModelDocument.h"
#endif

namespace WebCore {

using namespace HTMLNames;

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(DOMImplementation);

Ref<Document> DOMImplementation::protectedDocument()
{
    return m_document.get();
}

DOMImplementation::DOMImplementation(Document& document)
    : m_document(document)
{
}

ExceptionOr<Ref<DocumentType>> DOMImplementation::createDocumentType(const AtomString& qualifiedName, const String& publicId, const String& systemId)
{
    auto parseResult = Document::parseQualifiedName(qualifiedName);
    if (parseResult.hasException())
        return parseResult.releaseException();
    return DocumentType::create(protectedDocument(), qualifiedName, publicId, systemId);
}

static inline Ref<XMLDocument> createXMLDocument(const String& namespaceURI, const Settings& settings)
{
    RefPtr<XMLDocument> document;
    if (namespaceURI == SVGNames::svgNamespaceURI)
        document = SVGDocument::create(nullptr, settings, URL());
    else if (namespaceURI == HTMLNames::xhtmlNamespaceURI)
        document = XMLDocument::createXHTML(nullptr, settings, URL());
    else
        document = XMLDocument::create(nullptr, settings, URL());
    document->setParserContentPolicy({ ParserContentPolicy::AllowScriptingContent });
    return document.releaseNonNull();
}

ExceptionOr<Ref<XMLDocument>> DOMImplementation::createDocument(const AtomString& namespaceURI, const AtomString& qualifiedName, DocumentType* documentType)
{
    Ref document = createXMLDocument(namespaceURI, m_document->protectedSettings());
    document->setParserContentPolicy({ ParserContentPolicy::AllowScriptingContent });
    document->setContextDocument(m_document->contextDocument());
    document->setSecurityOriginPolicy(m_document->securityOriginPolicy());

    RefPtr<Element> documentElement;
    if (!qualifiedName.isEmpty()) {
        ASSERT(!document->domWindow()); // If domWindow is not null, createElementNS could find CustomElementRegistry and arbitrary scripts.
        auto result = document->createElementNS(namespaceURI, qualifiedName);
        if (result.hasException())
            return result.releaseException();
        documentElement = result.releaseReturnValue();
    }

    if (documentType)
        document->appendChild(*documentType);
    if (documentElement)
        document->appendChild(*documentElement);

    return document;
}

Ref<CSSStyleSheet> DOMImplementation::createCSSStyleSheet(const String&, const String& media)
{
    // FIXME: Title should be set.
    // FIXME: Media could have wrong syntax, in which case we should generate an exception.
    auto sheet = CSSStyleSheet::create(StyleSheetContents::create());
    sheet->setMediaQueries(MQ::MediaQueryParser::parse(media, { }));
    return sheet;
}

Ref<HTMLDocument> DOMImplementation::createHTMLDocument(String&& title)
{
    Ref document = HTMLDocument::create(nullptr, m_document->protectedSettings(), URL(), { });
    document->setParserContentPolicy({ ParserContentPolicy::AllowScriptingContent });
    document->open();
    document->write(nullptr, FixedVector<String> { "<!doctype html><html><head></head><body></body></html>"_s });
    if (!title.isNull()) {
        auto titleElement = HTMLTitleElement::create(titleTag, document);
        titleElement->appendChild(document->createTextNode(WTFMove(title)));
        ASSERT(document->head());
        document->protectedHead()->appendChild(titleElement);
    }
    document->setContextDocument(m_document->contextDocument());
    document->setSecurityOriginPolicy(m_document->securityOriginPolicy());
    return document;
}

Ref<Document> DOMImplementation::createDocument(const String& contentType, LocalFrame* frame, const Settings& settings, const URL& url, std::optional<ScriptExecutionContextIdentifier> documentIdentifier)
{
    // FIXME: Inelegant to have this here just because this is the home of DOM APIs for creating documents.
    // This is internal, not a DOM API. Maybe we should put it in a new class called DocumentFactory,
    // because of the analogy with HTMLElementFactory.

    // Plug-ins cannot take over for HTML, XHTML, plain text, or non-PDF images.
    if (equalLettersIgnoringASCIICase(contentType, "text/html"_s))
        return HTMLDocument::create(frame, settings, url, documentIdentifier);
    if (equalLettersIgnoringASCIICase(contentType, "application/xhtml+xml"_s))
        return XMLDocument::createXHTML(frame, settings, url);
    if (equalLettersIgnoringASCIICase(contentType, "text/plain"_s))
        return TextDocument::create(frame, settings, url, documentIdentifier);

#if ENABLE(PDFJS)
    if (frame && settings.pdfJSViewerEnabled() && MIMETypeRegistry::isPDFMIMEType(contentType))
        return PDFDocument::create(*frame, url);
#endif

    bool isImage = MIMETypeRegistry::isSupportedImageMIMEType(contentType);
    if (frame && isImage && !MIMETypeRegistry::isPDFMIMEType(contentType))
        return ImageDocument::create(*frame, url);

    // The "image documents for subframe PDFs" mode will override a PDF plug-in.
    if (frame && !frame->isMainFrame() && MIMETypeRegistry::isPDFMIMEType(contentType) && frame->settings().useImageDocumentForSubframePDF())
        return ImageDocument::create(*frame, url);

#if ENABLE(VIDEO)
    MediaEngineSupportParameters parameters;
    parameters.type = ContentType { contentType };
    parameters.url = url;
    if (MediaPlayer::supportsType(parameters) != MediaPlayer::SupportsType::IsNotSupported)
        return MediaDocument::create(frame, settings, url);
#endif

#if ENABLE(MODEL_ELEMENT)
    if (MIMETypeRegistry::isUSDMIMEType(contentType) && DeprecatedGlobalSettings::modelDocumentEnabled())
        return ModelDocument::create(frame, settings, url);
#endif

#if ENABLE(FTPDIR)
    if (equalLettersIgnoringASCIICase(contentType, "application/x-ftp-directory"_s))
        return FTPDirectoryDocument::create(frame, settings, url);
#endif

    // The following is the relatively costly lookup that requires initializing the plug-in database.
    if (frame && frame->page()) {
        if (frame->page()->pluginData().supportsWebVisibleMimeType(contentType, PluginData::OnlyApplicationPlugins))
            return PluginDocument::create(*frame, url);
    }

    // Items listed here, after the plug-in checks, can be overridden by plug-ins.
    // For example, plug-ins can take over support for PDF or SVG.
    if (frame && isImage)
        return ImageDocument::create(*frame, url);
    if (MIMETypeRegistry::isTextMIMEType(contentType))
        return TextDocument::create(frame, settings, url, documentIdentifier);
    if (equalLettersIgnoringASCIICase(contentType, "image/svg+xml"_s))
        return SVGDocument::create(frame, settings, url);
    if (MIMETypeRegistry::isXMLMIMEType(contentType)) {
        auto document = XMLDocument::create(frame, settings, url);
        document->overrideMIMEType(contentType);
        return document;
    }

    return HTMLDocument::create(frame, settings, url, documentIdentifier);
}

}
