/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 12, 2021.
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
#include "DOMParser.h"

#include "CommonAtomStrings.h"
#include "HTMLDocument.h"
#include "SVGDocument.h"
#include "SecurityOriginPolicy.h"
#include "TrustedType.h"
#include "XMLDocument.h"

namespace WebCore {

inline DOMParser::DOMParser(Document& contextDocument)
    : m_contextDocument(contextDocument)
    , m_settings(contextDocument.settings())
{
}

DOMParser::~DOMParser() = default;

Ref<DOMParser> DOMParser::create(Document& contextDocument)
{
    return adoptRef(*new DOMParser(contextDocument));
}

ExceptionOr<Ref<Document>> DOMParser::parseFromString(std::variant<RefPtr<TrustedHTML>, String>&& string, const AtomString& contentType)
{
    auto stringValueHolder = trustedTypeCompliantString(*m_contextDocument->scriptExecutionContext(), WTFMove(string), "DOMParser parseFromString"_s);

    if (stringValueHolder.hasException())
        return stringValueHolder.releaseException();

    RefPtr<Document> document;
    if (contentType == textHTMLContentTypeAtom())
        document = HTMLDocument::create(nullptr, m_settings, URL { });
    else if (contentType == applicationXHTMLContentTypeAtom())
        document = XMLDocument::createXHTML(nullptr, m_settings, URL { });
    else if (contentType == imageSVGContentTypeAtom())
        document = SVGDocument::create(nullptr, m_settings, URL { });
    else if (contentType == textXMLContentTypeAtom() || contentType == applicationXMLContentTypeAtom()) {
        document = XMLDocument::create(nullptr, m_settings, URL { });
        document->overrideMIMEType(contentType);
    } else
        return Exception { ExceptionCode::TypeError };

    if (m_contextDocument)
        document->setContextDocument(*m_contextDocument.get());
    document->setMarkupUnsafe(stringValueHolder.releaseReturnValue(), { });
    if (m_contextDocument) {
        document->setURL(m_contextDocument->url());
        document->setSecurityOriginPolicy(m_contextDocument->securityOriginPolicy());
    }
    return document.releaseNonNull();
}

} // namespace WebCore
