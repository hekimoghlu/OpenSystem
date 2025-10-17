/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 21, 2023.
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

#include "ExceptionOr.h"
#include "ScriptExecutionContextIdentifier.h"
#include "XMLDocument.h"
#include <wtf/WeakRef.h>

namespace WebCore {

class DOMImplementation final : public ScriptWrappable {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(DOMImplementation);
public:
    explicit DOMImplementation(Document&);

    void ref() { m_document->ref(); }
    void deref() { m_document->deref(); }
    Document& document() { return m_document; }

    WEBCORE_EXPORT ExceptionOr<Ref<DocumentType>> createDocumentType(const AtomString& qualifiedName, const String& publicId, const String& systemId);
    WEBCORE_EXPORT ExceptionOr<Ref<XMLDocument>> createDocument(const AtomString& namespaceURI, const AtomString& qualifiedName, DocumentType*);
    WEBCORE_EXPORT Ref<HTMLDocument> createHTMLDocument(String&& title);
    static bool hasFeature() { return true; }
    WEBCORE_EXPORT static Ref<CSSStyleSheet> createCSSStyleSheet(const String& title, const String& media);

    static Ref<Document> createDocument(const String& contentType, LocalFrame*, const Settings&, const URL&, std::optional<ScriptExecutionContextIdentifier> = std::nullopt);

private:
    Ref<Document> protectedDocument();

    WeakRef<Document, WeakPtrImplWithEventTargetData> m_document;
};

} // namespace WebCore
