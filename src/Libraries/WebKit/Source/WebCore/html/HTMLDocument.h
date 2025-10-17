/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 15, 2022.
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

#include "Document.h"
#include "TreeScopeOrderedMap.h"

namespace WebCore {

class HTMLDocument : public Document {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED_EXPORT(HTMLDocument, WEBCORE_EXPORT);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(HTMLDocument);
public:
    static Ref<HTMLDocument> create(LocalFrame*, const Settings&, const URL&, std::optional<ScriptExecutionContextIdentifier> = std::nullopt);
    static Ref<HTMLDocument> createSynthesizedDocument(LocalFrame&, const URL&);
    virtual ~HTMLDocument();
    
    std::optional<std::variant<RefPtr<WindowProxy>, RefPtr<Element>, RefPtr<HTMLCollection>>> namedItem(const AtomString&);
    Vector<AtomString> supportedPropertyNames() const;
    bool isSupportedPropertyName(const AtomString&) const;

    RefPtr<Element> documentNamedItem(const AtomString& name) const { return m_documentNamedItem.getElementByDocumentNamedItem(name, *this); }
    bool hasDocumentNamedItem(const AtomString& name) const { return m_documentNamedItem.contains(name); }
    bool documentNamedItemContainsMultipleElements(const AtomString& name) const { return m_documentNamedItem.containsMultiple(name); }
    void addDocumentNamedItem(const AtomString&, Element&);
    void removeDocumentNamedItem(const AtomString&, Element&);

    RefPtr<Element> windowNamedItem(const AtomString& name) const { return m_windowNamedItem.getElementByWindowNamedItem(name, *this); }
    bool hasWindowNamedItem(const AtomString& name) const { return m_windowNamedItem.contains(name); }
    bool windowNamedItemContainsMultipleElements(const AtomString& name) const { return m_windowNamedItem.containsMultiple(name); }
    void addWindowNamedItem(const AtomString&, Element&);
    void removeWindowNamedItem(const AtomString&, Element&);

    static bool isCaseSensitiveAttribute(const QualifiedName&);

protected:
    WEBCORE_EXPORT HTMLDocument(LocalFrame*, const Settings&, const URL&, std::optional<ScriptExecutionContextIdentifier>, DocumentClasses = { }, OptionSet<ConstructionFlag> = { });

private:
    bool isFrameSet() const final;
    Ref<DocumentParser> createParser() override;
    Ref<Document> cloneDocumentWithoutChildren() const final;

    TreeScopeOrderedMap m_documentNamedItem;
    TreeScopeOrderedMap m_windowNamedItem;
};

inline Ref<HTMLDocument> HTMLDocument::create(LocalFrame* frame, const Settings& settings, const URL& url, std::optional<ScriptExecutionContextIdentifier> identifier)
{
    auto document = adoptRef(*new HTMLDocument(frame, settings, url, identifier, { DocumentClass::HTML }));
    document->addToContextsMap();
    return document;
}

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_BEGIN(WebCore::HTMLDocument)
    static bool isType(const WebCore::Document& document) { return document.isHTMLDocument(); }
    static bool isType(const WebCore::Node& node)
    {
        auto* document = dynamicDowncast<WebCore::Document>(node);
        return document && isType(*document);
    }
SPECIALIZE_TYPE_TRAITS_END()
