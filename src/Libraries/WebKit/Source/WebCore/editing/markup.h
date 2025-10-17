/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 1, 2023.
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
#include "ExceptionOr.h"
#include "FloatSize.h"
#include "HTMLInterchange.h"
#include "MarkupExclusionRule.h"
#include "ParserContentPolicy.h"
#include "ShadowRoot.h"
#include <wtf/Forward.h>
#include <wtf/Function.h>
#include <wtf/HashMap.h>

namespace WebCore {

class ArchiveResource;
class ContainerNode;
class CustomElementRegistry;
class Document;
class DocumentFragment;
class Element;
class HTMLElement;
class LocalFrame;
class Node;
class Page;
class QualifiedName;
class VisibleSelection;

struct PresentationSize;
struct SimpleRange;

void replaceSubresourceURLs(Ref<DocumentFragment>&&, UncheckedKeyHashMap<AtomString, AtomString>&&);
void removeSubresourceURLAttributes(Ref<DocumentFragment>&&, Function<bool(const URL&)> shouldRemoveURL);

Ref<Page> createPageForSanitizingWebContent();
enum class MSOListQuirks : bool { CheckIfNeeded, Disabled };
String sanitizeMarkup(const String&, MSOListQuirks = MSOListQuirks::Disabled, std::optional<Function<void(DocumentFragment&)>> fragmentSanitizer = std::nullopt);
String sanitizedMarkupForFragmentInDocument(Ref<DocumentFragment>&&, Document&, MSOListQuirks, const String& originalMarkup);

class UserSelectNoneStateCache {
public:
    explicit UserSelectNoneStateCache(TreeType);

    bool nodeOnlyContainsUserSelectNone(Node& node) { return computeState(node) == State::OnlyUserSelectNone; }

private:
    ContainerNode* parentNode(Node&);
    Node* firstChild(Node&);
    Node* nextSibling(Node&);

    enum class State : uint8_t { NotUserSelectNone, Mixed, OnlyUserSelectNone };
    State computeState(Node&);

    UncheckedKeyHashMap<Ref<Node>, State> m_cache;
    bool m_useComposedTree;
};

WEBCORE_EXPORT Ref<DocumentFragment> createFragmentFromText(const SimpleRange& context, const String& text);
WEBCORE_EXPORT Ref<DocumentFragment> createFragmentFromMarkup(Document&, const String& markup, const String& baseURL, OptionSet<ParserContentPolicy> = { ParserContentPolicy::AllowScriptingContent });
ExceptionOr<Ref<DocumentFragment>> createFragmentForInnerOuterHTML(Element&, const String& markup, OptionSet<ParserContentPolicy>, CustomElementRegistry*);
RefPtr<DocumentFragment> createFragmentForTransformToFragment(Document&, String&& sourceString, const String& sourceMIMEType);
Ref<DocumentFragment> createFragmentForImageAndURL(Document&, const String&, PresentationSize preferredSize);
ExceptionOr<Ref<DocumentFragment>> createContextualFragment(Element&, const String& markup, OptionSet<ParserContentPolicy>);

bool isPlainTextMarkup(Node*);

// These methods are used by HTMLElement & ShadowRoot to replace the children with respected fragment/text.
ExceptionOr<void> replaceChildrenWithFragment(ContainerNode&, Ref<DocumentFragment>&&);

enum class ConvertBlocksToInlines : bool { No, Yes };
enum class SerializeComposedTree : bool { No, Yes };
enum class IgnoreUserSelectNone : bool { No, Yes };
enum class PreserveBaseElement : bool { No, Yes };
enum class PreserveDirectionForInlineText : bool { No, Yes };
WEBCORE_EXPORT String serializePreservingVisualAppearance(const SimpleRange&, Vector<Ref<Node>>* = nullptr, AnnotateForInterchange = AnnotateForInterchange::No, ConvertBlocksToInlines = ConvertBlocksToInlines::No, ResolveURLs = ResolveURLs::No);
String serializePreservingVisualAppearance(const VisibleSelection&, ResolveURLs = ResolveURLs::No, SerializeComposedTree = SerializeComposedTree::No,
    IgnoreUserSelectNone = IgnoreUserSelectNone::Yes, PreserveBaseElement = PreserveBaseElement::No, PreserveDirectionForInlineText = PreserveDirectionForInlineText::No, Vector<Ref<Node>>* = nullptr);

enum class SerializedNodes : uint8_t { SubtreeIncludingNode, SubtreesOfChildren };
enum class SerializationSyntax : uint8_t { HTML, XML };
enum class SerializeShadowRoots : uint8_t { Explicit, Serializable, AllForInterchange };
WEBCORE_EXPORT String serializeFragment(const Node&, SerializedNodes, Vector<Ref<Node>>* = nullptr, ResolveURLs = ResolveURLs::No, std::optional<SerializationSyntax> = std::nullopt, SerializeShadowRoots = SerializeShadowRoots::Explicit, Vector<Ref<ShadowRoot>>&& explicitShadowRoots = { }, const Vector<MarkupExclusionRule>& exclusionRules = { });
WEBCORE_EXPORT String serializeFragmentWithURLReplacement(const Node&, SerializedNodes, Vector<Ref<Node>>*, ResolveURLs, std::optional<SerializationSyntax>, UncheckedKeyHashMap<String, String>&& replacementURLStrings, UncheckedKeyHashMap<RefPtr<CSSStyleSheet>, String>&& replacementURLStringsForCSSStyleSheet, SerializeShadowRoots = SerializeShadowRoots::Explicit, Vector<Ref<ShadowRoot>>&& explicitShadowRoots = { }, const Vector<MarkupExclusionRule>& exclusionRules = { });

String urlToMarkup(const URL&, const String& title);

WEBCORE_EXPORT String documentTypeString(const Document&);

}
