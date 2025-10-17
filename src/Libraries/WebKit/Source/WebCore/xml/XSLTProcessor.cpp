/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 16, 2025.
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
#include "XSLTProcessor.h"

#if ENABLE(XSLT)

#include "DOMImplementation.h"
#include "CachedResourceLoader.h"
#include "CommonAtomStrings.h"
#include "ContentSecurityPolicy.h"
#include "DocumentFragment.h"
#include "FrameLoader.h"
#include "LocalFrame.h"
#include "LocalFrameView.h"
#include "SecurityOrigin.h"
#include "SecurityOriginPolicy.h"
#include "Text.h"
#include "TextResourceDecoder.h"
#include "XMLDocument.h"
#include "markup.h"
#include <wtf/Assertions.h>
#include <wtf/text/MakeString.h>

namespace WebCore {

static inline void transformTextStringToXHTMLDocumentString(String& text)
{
    // Modify the output so that it is a well-formed XHTML document with a <pre> tag enclosing the text.
    text = makeStringByReplacingAll(text, '&', "&amp;"_s);
    text = makeStringByReplacingAll(text, '<', "&lt;"_s);
    text = makeString("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"
        "<!DOCTYPE html PUBLIC \"-//W3C//DTD XHTML 1.0 Strict//EN\" \"http://www.w3.org/TR/xhtml1/DTD/xhtml1-strict.dtd\">\n"
        "<html xmlns=\"http://www.w3.org/1999/xhtml\">\n"
        "<head><title/></head>\n"
        "<body>\n"
        "<pre>"_s, text, "</pre>\n"
        "</body>\n"
        "</html>\n"_s);
}

XSLTProcessor::~XSLTProcessor()
{
    // Stylesheet shouldn't outlive its root node.
    ASSERT(!m_stylesheetRootNode || !m_stylesheet || m_stylesheet->hasOneRef());
}

Ref<Document> XSLTProcessor::createDocumentFromSource(const String& sourceString,
    const String& sourceEncoding, const String& sourceMIMEType, Node* sourceNode, LocalFrame* frame)
{
    Ref<Document> ownerDocument(sourceNode->document());
    bool sourceIsDocument = (sourceNode == &ownerDocument.get());
    String documentSource = sourceString;

    RefPtr<Document> result;
    if (sourceMIMEType == textPlainContentTypeAtom()) {
        result = XMLDocument::createXHTML(frame, ownerDocument->settings(), sourceIsDocument ? ownerDocument->url() : URL());
        transformTextStringToXHTMLDocumentString(documentSource);
    } else
        result = DOMImplementation::createDocument(sourceMIMEType, frame, ownerDocument->settings(), sourceIsDocument ? ownerDocument->url() : URL());

    // Before parsing, we need to save & detach the old document and get the new document
    // in place. We have to do this only if we're rendering the result document.
    if (frame) {
        if (auto* view = frame->view())
            view->clear();

        if (Document* oldDocument = frame->document()) {
            result->setTransformSourceDocument(oldDocument);
            result->takeDOMWindowFrom(*oldDocument);
            result->setSecurityOriginPolicy(oldDocument->securityOriginPolicy());
            result->setCookieURL(oldDocument->cookieURL());
            result->setFirstPartyForCookies(oldDocument->firstPartyForCookies());
            result->setSiteForCookies(oldDocument->siteForCookies());
            result->setStrictMixedContentMode(oldDocument->isStrictMixedContentMode());
            CheckedRef resultCSP = *result->contentSecurityPolicy();
            CheckedRef oldDocumentCSP = *oldDocument->contentSecurityPolicy();
            resultCSP->copyStateFrom(oldDocumentCSP.ptr());
            resultCSP->copyUpgradeInsecureRequestStateFrom(oldDocumentCSP);
        }

        frame->setDocument(result.copyRef());
    }

    auto decoder = TextResourceDecoder::create(sourceMIMEType);
    decoder->setEncoding(sourceEncoding.isEmpty() ? PAL::UTF8Encoding() : PAL::TextEncoding(sourceEncoding), TextResourceDecoder::EncodingFromXMLHeader);
    result->setDecoder(WTFMove(decoder));

    result->setMarkupUnsafe(documentSource, { ParserContentPolicy::AllowDeclarativeShadowRoots });

    return result.releaseNonNull();
}

RefPtr<Document> XSLTProcessor::transformToDocument(Node& sourceNode)
{
    String resultMIMEType;
    String resultString;
    String resultEncoding;
    if (!transformToString(sourceNode, resultMIMEType, resultString, resultEncoding))
        return nullptr;
    return createDocumentFromSource(resultString, resultEncoding, resultMIMEType, &sourceNode, nullptr);
}

RefPtr<DocumentFragment> XSLTProcessor::transformToFragment(Node& sourceNode, Document& outputDocument)
{
    String resultMIMEType;
    String resultString;
    String resultEncoding;

    // If the output document is HTML, default to HTML method.
    if (outputDocument.isHTMLDocument())
        resultMIMEType = textHTMLContentTypeAtom();

    if (!transformToString(sourceNode, resultMIMEType, resultString, resultEncoding))
        return nullptr;
    return createFragmentForTransformToFragment(outputDocument, WTFMove(resultString), WTFMove(resultMIMEType));
}

void XSLTProcessor::setParameter(const String& /*namespaceURI*/, const String& localName, const String& value)
{
    if (localName.isNull() || value.isNull())
        return;

    // FIXME: namespace support?
    // should make a QualifiedName here but we'd have to expose the impl
    m_parameters.set(localName, value);
}

String XSLTProcessor::getParameter(const String& /*namespaceURI*/, const String& localName) const
{
    if (localName.isNull())
        return { };

    // FIXME: namespace support?
    // should make a QualifiedName here but we'd have to expose the impl
    return m_parameters.get(localName);
}

void XSLTProcessor::removeParameter(const String& /*namespaceURI*/, const String& localName)
{
    if (localName.isNull())
        return;

    // FIXME: namespace support?
    m_parameters.remove(localName);
}

void XSLTProcessor::reset()
{
    m_stylesheet = nullptr;
    m_stylesheetRootNode = nullptr;
    m_parameters.clear();
}

} // namespace WebCore

#endif // ENABLE(XSLT)
