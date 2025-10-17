/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 14, 2022.
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

#include "CSSStyleSheet.h"
#include "Element.h"
#include "markup.h"
#include <wtf/HashMap.h>
#include <wtf/text/StringBuilder.h>

namespace WebCore {

class Attribute;
class DocumentType;
class Element;
class LocalFrame;
class Node;
class Range;
class ShadowRoot;

typedef UncheckedKeyHashMap<AtomString, AtomStringImpl*> Namespaces;

enum class EntityMask : uint8_t {
    Amp = 1 << 0,
    Lt = 1 << 1,
    Gt = 1 << 2,
    Quot = 1 << 3,
    Nbsp = 1 << 4,
    Tab = 1 << 5,
    LineFeed = 1 << 6,
    CarriageReturn = 1 << 7,

    // Non-breaking space needs to be escaped in innerHTML for compatibility reason. See http://trac.webkit.org/changeset/32879
    // However, we cannot do this in a XML document because it does not have the entity reference defined (See the bug 19215).
};

constexpr OptionSet<EntityMask> EntityMaskInCDATA = { };
constexpr OptionSet<EntityMask> EntityMaskInPCDATA = { EntityMask::Amp, EntityMask::Lt, EntityMask::Gt };
constexpr auto EntityMaskInHTMLPCDATA = EntityMaskInPCDATA | EntityMask::Nbsp;
constexpr OptionSet<EntityMask> EntityMaskInAttributeValue = { EntityMask::Amp, EntityMask::Lt, EntityMask::Gt,
    EntityMask::Quot, EntityMask::Tab, EntityMask::LineFeed, EntityMask::CarriageReturn };
constexpr auto EntityMaskInHTMLAttributeValue = { EntityMask::Amp, EntityMask::Quot, EntityMask::Nbsp };

class MarkupAccumulator {
    WTF_MAKE_NONCOPYABLE(MarkupAccumulator);
public:
    MarkupAccumulator(Vector<Ref<Node>>*, ResolveURLs, SerializationSyntax, SerializeShadowRoots = SerializeShadowRoots::Explicit, Vector<Ref<ShadowRoot>>&& explicitShadowRoots = { }, const Vector<MarkupExclusionRule>& exclusionRules = { });
    virtual ~MarkupAccumulator();

    String serializeNodes(Node& targetNode, SerializedNodes);

    static void appendCharactersReplacingEntities(StringBuilder&, const String&, unsigned, unsigned, OptionSet<EntityMask>);
    void enableURLReplacement(UncheckedKeyHashMap<String, String>&& replacementURLStrings, UncheckedKeyHashMap<RefPtr<CSSStyleSheet>, String>&& replacementURLStringsForCSSStyleSheet);

protected:
    unsigned length() const { return m_markup.length(); }
    bool containsOnlyASCII() const { return m_markup.containsOnlyASCII(); }

    StringBuilder takeMarkup();

    template<typename ...StringTypes> void append(StringTypes&&... strings) { m_markup.append(std::forward<StringTypes>(strings)...); }

    void startAppendingNode(const Node&, Namespaces* = nullptr);
    void endAppendingNode(const Node&);

    virtual void appendStartTag(StringBuilder&, const Element&, Namespaces*);
    virtual void appendEndTag(StringBuilder&, const Element&);
    virtual void appendCustomAttributes(StringBuilder&, const Element&, Namespaces*);
    virtual void appendText(StringBuilder&, const Text&);
    virtual bool appendContentsForNode(StringBuilder& result, const Node&);

    void appendOpenTag(StringBuilder&, const Element&, Namespaces*);
    void appendCloseTag(StringBuilder&, const Element&);

    void appendNonElementNode(StringBuilder&, const Node&, Namespaces*);

    static void appendAttributeValue(StringBuilder&, const String&, bool isSerializingHTML);
    bool appendAttribute(StringBuilder&, const Element&, const Attribute&, Namespaces*);

    OptionSet<EntityMask> entityMaskForText(const Text&) const;

    Vector<Ref<Node>>* const m_nodes;

private:
    void appendNamespace(StringBuilder&, const AtomString& prefix, const AtomString& namespaceURI, Namespaces&, bool allowEmptyDefaultNS = false);
    enum class IsCreatedByURLReplacement : bool { No, Yes };
    std::pair<String, IsCreatedByURLReplacement> resolveURLIfNeeded(const Element&, const String&) const;
    bool shouldIncludeShadowRoots() const;
    bool includeShadowRoot(const ShadowRoot&) const;
    void serializeNodesWithNamespaces(Node& targetNode, SerializedNodes, const Namespaces*);
    bool inXMLFragmentSerialization() const { return m_serializationSyntax == SerializationSyntax::XML; }
    void generateUniquePrefix(QualifiedName&, const Namespaces&);
    QualifiedName xmlAttributeSerialization(const Attribute&, Namespaces*);
    LocalFrame* frameForAttributeReplacement(const Element&) const;
    Attribute replaceAttributeIfNecessary(const Element&, const Attribute&);
    bool appendURLAttributeForReplacementIfNecessary(StringBuilder&, const Element&, Namespaces*);
    const ShadowRoot* suitableShadowRoot(const Node&);
    bool shouldExcludeElement(const Element&);
    void appendStartTagWithURLReplacement(StringBuilder&, const Element&, Namespaces*);

    StringBuilder m_markup;
    const ResolveURLs m_resolveURLs;
    const SerializationSyntax m_serializationSyntax;
    unsigned m_prefixLevel { 0 };
    UncheckedKeyHashMap<String, String> m_replacementURLStrings;
    UncheckedKeyHashMap<RefPtr<CSSStyleSheet>, String> m_replacementURLStringsForCSSStyleSheet;
    SerializeShadowRoots m_serializeShadowRoots;
    Vector<Ref<ShadowRoot>> m_explicitShadowRoots;
    Vector<MarkupExclusionRule> m_exclusionRules;
    struct URLReplacementData {
        UncheckedKeyHashMap<String, String> replacementURLStrings;
        UncheckedKeyHashMap<RefPtr<CSSStyleSheet>, String> replacementURLStringsForCSSStyleSheet;
    };
    std::optional<URLReplacementData> m_urlReplacementData;
};

inline void MarkupAccumulator::endAppendingNode(const Node& node)
{
    if (RefPtr element = dynamicDowncast<Element>(node))
        appendEndTag(m_markup, *element);
    else if (suitableShadowRoot(node))
        m_markup.append("</template>"_s);
}

} // namespace WebCore
