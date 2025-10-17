/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 13, 2025.
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
#include "XMLErrors.h"

#include "Document.h"
#include "HTMLBodyElement.h"
#include "HTMLDivElement.h"
#include "HTMLHeadElement.h"
#include "HTMLHeadingElement.h"
#include "HTMLHtmlElement.h"
#include "HTMLNames.h"
#include "HTMLParagraphElement.h"
#include "HTMLStyleElement.h"
#include "LocalFrame.h"
#include "SVGNames.h"
#include "Text.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(XMLErrors);

using namespace HTMLNames;

const int maxErrors = 25;

XMLErrors::XMLErrors(Document& document)
    : m_document(document)
{
}

void XMLErrors::handleError(Type type, const char* message, int lineNumber, int columnNumber)
{
    handleError(type, message, TextPosition(OrdinalNumber::fromOneBasedInt(lineNumber), OrdinalNumber::fromOneBasedInt(columnNumber)));
}

void XMLErrors::handleError(Type type, const char* message, TextPosition position)
{
    if (type == Type::Fatal || (m_errorCount < maxErrors && (!m_lastErrorPosition || (m_lastErrorPosition->m_line != position.m_line && m_lastErrorPosition->m_column != position.m_column)))) {
        switch (type) {
        case Type::Warning:
            appendErrorMessage("warning"_s, position, message);
            break;
        case Type::Fatal:
        case Type::NonFatal:
            appendErrorMessage("error"_s, position, message);
        }

        m_lastErrorPosition = position;
        ++m_errorCount;
    }
}

void XMLErrors::appendErrorMessage(ASCIILiteral typeString, TextPosition position, const char* message)
{
    // <typeString> on line <lineNumber> at column <columnNumber>: <message>
    m_errorMessages.append(typeString, " on line "_s, position.m_line.oneBasedInt(), " at column "_s, position.m_column.oneBasedInt(), ": "_s, unsafeSpan(message));
}

static inline Ref<Element> createXHTMLParserErrorHeader(Document& document, String&& errorMessages)
{
    Ref reportElement = document.createElement(QualifiedName(nullAtom(), "parsererror"_s, xhtmlNamespaceURI), true);

    Attribute reportAttribute(styleAttr, "display: block; white-space: pre; border: 2px solid #c77; padding: 0 1em 0 1em; margin: 1em; background-color: #fdd; color: black"_s);
    reportElement->parserSetAttributes(std::span(&reportAttribute, 1));

    Ref h3 = HTMLHeadingElement::create(h3Tag, document);
    reportElement->parserAppendChild(h3);
    h3->parserAppendChild(Text::create(document, "This page contains the following errors:"_s));

    Ref fixed = HTMLDivElement::create(document);
    Attribute fixedAttribute(styleAttr, "font-family:monospace;font-size:12px"_s);
    fixed->parserSetAttributes(std::span(&fixedAttribute, 1));
    reportElement->parserAppendChild(fixed);

    fixed->parserAppendChild(Text::create(document, WTFMove(errorMessages)));

    h3 = HTMLHeadingElement::create(h3Tag, document);
    reportElement->parserAppendChild(h3);
    h3->parserAppendChild(Text::create(document, "Below is a rendering of the page up to the first error."_s));

    return reportElement;
}

void XMLErrors::insertErrorMessageBlock()
{
    // One or more errors occurred during parsing of the code. Display an error block to the user above
    // the normal content (the DOM tree is created manually and includes line/col info regarding
    // where the errors are located)

    // Create elements for display
    Ref document = m_document.get();
    RefPtr documentElement = document->documentElement();
    if (!documentElement) {
        Ref rootElement = HTMLHtmlElement::create(document);
        Ref body = HTMLBodyElement::create(document);
        rootElement->parserAppendChild(body);
        document->parserAppendChild(WTFMove(rootElement));
        documentElement = WTFMove(body);
    } else if (documentElement->namespaceURI() == SVGNames::svgNamespaceURI) {
        Ref rootElement = HTMLHtmlElement::create(document);
        Ref head = HTMLHeadElement::create(document);
        Ref style = HTMLStyleElement::create(document);
        head->parserAppendChild(style);
        style->parserAppendChild(document->createTextNode("html, body { height: 100% } parsererror + svg { width: 100%; height: 100% }"_s));
        style->finishParsingChildren();
        rootElement->parserAppendChild(WTFMove(head));
        Ref body = HTMLBodyElement::create(document);
        rootElement->parserAppendChild(body);

        document->parserRemoveChild(*documentElement);
        if (!documentElement->parentNode())
            body->parserAppendChild(*documentElement);

        document->parserAppendChild(WTFMove(rootElement));

        documentElement = WTFMove(body);
    }

    Ref reportElement = createXHTMLParserErrorHeader(document, m_errorMessages.toString());

#if ENABLE(XSLT)
    if (document->transformSourceDocument()) {
        Attribute attribute(styleAttr, "white-space: normal"_s);
        Ref paragraph = HTMLParagraphElement::create(document);
        paragraph->parserSetAttributes(std::span(&attribute, 1));
        paragraph->parserAppendChild(document->createTextNode("This document was created as the result of an XSL transformation. The line and column numbers given are from the transformed result."_s));
        reportElement->parserAppendChild(WTFMove(paragraph));
    }
#endif

    if (RefPtr firstChild = documentElement->firstChild())
        documentElement->parserInsertBefore(WTFMove(reportElement), firstChild.releaseNonNull());
    else
        documentElement->parserAppendChild(WTFMove(reportElement));

    document->updateStyleIfNeeded();
}

} // namespace WebCore
