/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 11, 2023.
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

#include "CSSSelector.h"
#include "ContentSecurityPolicy.h"
#include "InspectorStyleSheet.h"
#include "InspectorWebAgentBase.h"
#include "SecurityContext.h"
#include "Timer.h"
#include <JavaScriptCore/InspectorBackendDispatchers.h>
#include <wtf/CheckedPtr.h>
#include <wtf/HashMap.h>
#include <wtf/HashSet.h>
#include <wtf/JSONValues.h>
#include <wtf/OptionSet.h>
#include <wtf/RefPtr.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/WeakHashMap.h>
#include <wtf/WeakHashSet.h>
#include <wtf/text/WTFString.h>

namespace Inspector {
class CSSFrontendDispatcher;
}

namespace WebCore {

class CSSRule;
class CSSStyleRule;
class CSSStyleSheet;
class Document;
class Element;
class EventTarget;
class Node;
class NodeList;
class RenderObject;
class Settings;
class StyleRule;
class WeakPtrImplWithEventTargetData;

namespace Style {
class Resolver;
}

class InspectorCSSAgent final : public InspectorAgentBase , public Inspector::CSSBackendDispatcherHandler , public InspectorStyleSheet::Listener, public CanMakeCheckedPtr<InspectorCSSAgent> {
    WTF_MAKE_NONCOPYABLE(InspectorCSSAgent);
    WTF_MAKE_TZONE_ALLOCATED(InspectorCSSAgent);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(InspectorCSSAgent);
public:
    explicit InspectorCSSAgent(PageAgentContext&);
    ~InspectorCSSAgent();

    class InlineStyleOverrideScope {
    public:
        InlineStyleOverrideScope(SecurityContext& context)
            : m_contentSecurityPolicy(context.contentSecurityPolicy())
        {
            m_contentSecurityPolicy->setOverrideAllowInlineStyle(true);
        }

        ~InlineStyleOverrideScope()
        {
            m_contentSecurityPolicy->setOverrideAllowInlineStyle(false);
        }

    private:
        ContentSecurityPolicy* m_contentSecurityPolicy;
    };

    static std::optional<Inspector::Protocol::CSS::PseudoId> protocolValueForPseudoId(PseudoId);

    // InspectorAgentBase
    void didCreateFrontendAndBackend(Inspector::FrontendRouter*, Inspector::BackendDispatcher*);
    void willDestroyFrontendAndBackend(Inspector::DisconnectReason);

    // CSSBackendDispatcherHandler
    Inspector::Protocol::ErrorStringOr<void> enable();
    Inspector::Protocol::ErrorStringOr<void> disable();
    Inspector::Protocol::ErrorStringOr<Ref<JSON::ArrayOf<Inspector::Protocol::CSS::CSSComputedStyleProperty>>> getComputedStyleForNode(Inspector::Protocol::DOM::NodeId);
    Inspector::Protocol::ErrorStringOr<Ref<Inspector::Protocol::CSS::Font>> getFontDataForNode(Inspector::Protocol::DOM::NodeId);
    Inspector::Protocol::ErrorStringOr<std::tuple<RefPtr<Inspector::Protocol::CSS::CSSStyle> /* inlineStyle */, RefPtr<Inspector::Protocol::CSS::CSSStyle> /* attributesStyle */>> getInlineStylesForNode(Inspector::Protocol::DOM::NodeId);
    Inspector::Protocol::ErrorStringOr<std::tuple<RefPtr<JSON::ArrayOf<Inspector::Protocol::CSS::RuleMatch>>, RefPtr<JSON::ArrayOf<Inspector::Protocol::CSS::PseudoIdMatches>>, RefPtr<JSON::ArrayOf<Inspector::Protocol::CSS::InheritedStyleEntry>>>> getMatchedStylesForNode(Inspector::Protocol::DOM::NodeId, std::optional<bool>&& includePseudo, std::optional<bool>&& includeInherited);
    Inspector::Protocol::ErrorStringOr<Ref<JSON::ArrayOf<Inspector::Protocol::CSS::CSSStyleSheetHeader>>> getAllStyleSheets();
    Inspector::Protocol::ErrorStringOr<Ref<Inspector::Protocol::CSS::CSSStyleSheetBody>> getStyleSheet(const Inspector::Protocol::CSS::StyleSheetId&);
    Inspector::Protocol::ErrorStringOr<String> getStyleSheetText(const Inspector::Protocol::CSS::StyleSheetId&);
    Inspector::Protocol::ErrorStringOr<void> setStyleSheetText(const Inspector::Protocol::CSS::StyleSheetId&, const String& text);
    Inspector::Protocol::ErrorStringOr<Ref<Inspector::Protocol::CSS::CSSStyle>> setStyleText(Ref<JSON::Object>&& styleId, const String& text);
    Inspector::Protocol::ErrorStringOr<Ref<Inspector::Protocol::CSS::CSSRule>> setRuleSelector(Ref<JSON::Object>&& ruleId, const String& selector);
    Inspector::Protocol::ErrorStringOr<Ref<Inspector::Protocol::CSS::Grouping>> setGroupingHeaderText(Ref<JSON::Object>&& ruleId, const String& headerText);
    Inspector::Protocol::ErrorStringOr<Inspector::Protocol::CSS::StyleSheetId> createStyleSheet(const Inspector::Protocol::Network::FrameId&);
    Inspector::Protocol::ErrorStringOr<Ref<Inspector::Protocol::CSS::CSSRule>> addRule(const Inspector::Protocol::CSS::StyleSheetId&, const String& selector);
    Inspector::Protocol::ErrorStringOr<Ref<JSON::ArrayOf<Inspector::Protocol::CSS::CSSPropertyInfo>>> getSupportedCSSProperties();
    Inspector::Protocol::ErrorStringOr<Ref<JSON::ArrayOf<String>>> getSupportedSystemFontFamilyNames();
    Inspector::Protocol::ErrorStringOr<void> forcePseudoState(Inspector::Protocol::DOM::NodeId, Ref<JSON::Array>&& forcedPseudoClasses);
    Inspector::Protocol::ErrorStringOr<void> setLayoutContextTypeChangedMode(Inspector::Protocol::CSS::LayoutContextTypeChangedMode);

    // InspectorStyleSheet::Listener
    void styleSheetChanged(InspectorStyleSheet*);

    // InspectorInstrumentation
    void documentDetached(Document&);
    void mediaQueryResultChanged();
    void activeStyleSheetsUpdated(Document&);
    bool forcePseudoState(const Element&, CSSSelector::PseudoClass);
    void didChangeRendererForDOMNode(Node&);
    void didAddEventListener(EventTarget&);
    void willRemoveEventListener(EventTarget&);

    // InspectorDOMAgent hooks
    void didRemoveDOMNode(Node&, Inspector::Protocol::DOM::NodeId);
    void didModifyDOMAttr(Element&);

    enum class LayoutFlag : uint8_t {
        Rendered = 1 << 0,
        Flex = 1 << 1,
        Grid = 1 << 2,
        Event = 1 << 3,
        Scrollable = 1 << 4,
    };
    OptionSet<LayoutFlag> layoutFlagsForNode(Node&);
    RefPtr<JSON::ArrayOf<String /* Inspector::Protocol::CSS::LayoutFlag */>> protocolLayoutFlagsForNode(Node&);

    void reset();

private:
    class StyleSheetAction;
    class SetStyleSheetTextAction;
    class SetStyleTextAction;
    class SetRuleHeaderTextAction;
    class AddRuleAction;

    using IdToInspectorStyleSheet = HashMap<Inspector::Protocol::CSS::StyleSheetId, RefPtr<InspectorStyleSheet>>;
    using CSSStyleSheetToInspectorStyleSheet = HashMap<CSSStyleSheet*, RefPtr<InspectorStyleSheet>>;
    using DocumentToViaInspectorStyleSheet = HashMap<RefPtr<Document>, Vector<RefPtr<InspectorStyleSheet>>>; // "via inspector" stylesheets
    using PseudoClassHashSet = HashSet<CSSSelector::PseudoClass, IntHash<CSSSelector::PseudoClass>, WTF::StrongEnumHashTraits<CSSSelector::PseudoClass>>;

    InspectorStyleSheetForInlineStyle& asInspectorStyleSheet(StyledElement&);
    Element* elementForId(Inspector::Protocol::ErrorString&, Inspector::Protocol::DOM::NodeId);
    Node* nodeForId(Inspector::Protocol::ErrorString&, Inspector::Protocol::DOM::NodeId);

    void collectAllStyleSheets(Vector<InspectorStyleSheet*>&);
    void collectAllDocumentStyleSheets(Document&, Vector<CSSStyleSheet*>&);
    void collectStyleSheets(CSSStyleSheet*, Vector<CSSStyleSheet*>&);
    void setActiveStyleSheetsForDocument(Document&, Vector<CSSStyleSheet*>& activeStyleSheets);

    Inspector::Protocol::CSS::StyleSheetId unbindStyleSheet(InspectorStyleSheet*);
    InspectorStyleSheet* bindStyleSheet(CSSStyleSheet*);
    InspectorStyleSheet* assertStyleSheetForId(Inspector::Protocol::ErrorString&, const Inspector::Protocol::CSS::StyleSheetId&);
    InspectorStyleSheet* createInspectorStyleSheetForDocument(Document&);
    Inspector::Protocol::CSS::StyleSheetOrigin detectOrigin(CSSStyleSheet* pageStyleSheet, Document* ownerDocument);

    RefPtr<Inspector::Protocol::CSS::CSSRule> buildObjectForRule(const StyleRule*, Style::Resolver&, Element&);
    RefPtr<Inspector::Protocol::CSS::CSSRule> buildObjectForRule(CSSStyleRule*);
    Ref<JSON::ArrayOf<Inspector::Protocol::CSS::RuleMatch>> buildArrayForMatchedRuleList(const Vector<RefPtr<const StyleRule>>&, Style::Resolver&, Element&, PseudoId);
    RefPtr<Inspector::Protocol::CSS::CSSStyle> buildObjectForAttributesStyle(StyledElement&);

    void nodeHasLayoutFlagsChange(Node&);
    void nodesWithPendingLayoutFlagsChangeDispatchTimerFired();

    void resetPseudoStates();

    std::unique_ptr<Inspector::CSSFrontendDispatcher> m_frontendDispatcher;
    RefPtr<Inspector::CSSBackendDispatcher> m_backendDispatcher;

    WeakRef<Page> m_inspectedPage;
    IdToInspectorStyleSheet m_idToInspectorStyleSheet;
    CSSStyleSheetToInspectorStyleSheet m_cssStyleSheetToInspectorStyleSheet;
    HashMap<Node*, Ref<InspectorStyleSheetForInlineStyle>> m_nodeToInspectorStyleSheet; // bogus "stylesheets" with elements' inline styles
    DocumentToViaInspectorStyleSheet m_documentToInspectorStyleSheet;
    HashMap<Document*, HashSet<CSSStyleSheet*>> m_documentToKnownCSSStyleSheets;
    HashMap<Inspector::Protocol::DOM::NodeId, PseudoClassHashSet> m_nodeIdToForcedPseudoState;
    HashSet<Document*> m_documentsWithForcedPseudoStates;

    int m_lastStyleSheetId { 1 };
    bool m_creatingViaInspectorStyleSheet { false };

    WeakHashMap<Node, OptionSet<LayoutFlag>, WeakPtrImplWithEventTargetData> m_lastLayoutFlagsForNode;
    WeakHashSet<Node, WeakPtrImplWithEventTargetData> m_nodesWithPendingLayoutFlagsChange;
    Timer m_nodesWithPendingLayoutFlagsChangeDispatchTimer;

    Inspector::Protocol::CSS::LayoutContextTypeChangedMode m_layoutContextTypeChangedMode { Inspector::Protocol::CSS::LayoutContextTypeChangedMode::Observed };
};

} // namespace WebCore
