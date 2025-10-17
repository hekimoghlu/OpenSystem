/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 23, 2024.
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

#include "Position.h"
#include "QualifiedName.h"
#include "TextManipulationControllerExclusionRule.h"
#include "TextManipulationControllerManipulationFailure.h"
#include "TextManipulationItem.h"
#include <wtf/CheckedRef.h>
#include <wtf/CompletionHandler.h>
#include <wtf/Markable.h>
#include <wtf/ObjectIdentifier.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/WeakHashSet.h>
#include <wtf/WeakPtr.h>

namespace WebCore {

class Document;
class Element;
class VisiblePosition;

using NodeSet = UncheckedKeyHashSet<Ref<Node>>;

class TextManipulationController final : public CanMakeWeakPtr<TextManipulationController>, public CanMakeCheckedPtr<TextManipulationController> {
    WTF_MAKE_TZONE_ALLOCATED(TextManipulationController);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(TextManipulationController);
public:
    TextManipulationController(Document&);
    
    using ManipulationFailure = TextManipulationControllerManipulationFailure;
    using ExclusionRule = TextManipulationControllerExclusionRule;

    using ManipulationItemCallback = Function<void(Document&, const Vector<TextManipulationItem>&)>;
    WEBCORE_EXPORT void startObservingParagraphs(ManipulationItemCallback&&, Vector<ExclusionRule>&& = { });

    void didUpdateContentForNode(Node&);
    void didAddOrCreateRendererForNode(Node&);
    void removeNode(Node&);

    WEBCORE_EXPORT Vector<ManipulationFailure> completeManipulation(const Vector<TextManipulationItem>&);

private:
    void observeParagraphs(const Position& start, const Position& end);
    void scheduleObservationUpdate();

    struct ManipulationItemData {
        Position start;
        Position end;

        WeakPtr<Element, WeakPtrImplWithEventTargetData> element;
        QualifiedName attributeName { nullQName() };

        Vector<TextManipulationToken> tokens;
    };

    struct ManipulationUnit {
        Ref<Node> node;
        Vector<TextManipulationToken> tokens;
        bool areAllTokensExcluded { true };
        bool firstTokenContainsDelimiter { false };
        bool lastTokenContainsDelimiter { false };
    };
    ManipulationUnit createUnit(const Vector<String>&, Node&);
    void parse(ManipulationUnit&, const String&, Node&);

    bool shouldExcludeNodeBasedOnStyle(const Node&);

    void addItem(ManipulationItemData&&);
    void addItemIfPossible(Vector<ManipulationUnit>&&);
    void flushPendingItemsForCallback();

    enum class IsNodeManipulated : bool { No, Yes };
    struct NodeInsertion {
        RefPtr<Node> parentIfDifferentFromCommonAncestor;
        Ref<Node> child;
        IsNodeManipulated isChildManipulated { IsNodeManipulated::Yes };
    };
    using NodeEntry = std::pair<Ref<Node>, Ref<Node>>;
    Vector<Ref<Node>> getPath(Node*, Node*);
    void updateInsertions(Vector<NodeEntry>&, const Vector<Ref<Node>>&, Node*, NodeSet&, Vector<NodeInsertion>&);
    std::optional<ManipulationFailure::Type> replace(const ManipulationItemData&, const Vector<TextManipulationToken>&, NodeSet& containersWithoutVisualOverflowBeforeReplacement);

    WeakPtr<Document, WeakPtrImplWithEventTargetData> m_document;
    WeakHashSet<Node, WeakPtrImplWithEventTargetData> m_manipulatedNodes;
    WeakHashSet<Node, WeakPtrImplWithEventTargetData> m_manipulatedNodesWithNewContent;
    WeakHashSet<Node, WeakPtrImplWithEventTargetData> m_addedOrNewlyRenderedNodes;

    UncheckedKeyHashMap<String, bool> m_cachedFontFamilyExclusionResults;

    bool m_didScheduleObservationUpdate { false };

    ManipulationItemCallback m_callback;
    Vector<TextManipulationItem> m_pendingItemsForCallback;

    Vector<ExclusionRule> m_exclusionRules;
    UncheckedKeyHashMap<TextManipulationItemIdentifier, ManipulationItemData> m_items;
};

} // namespace WebCore
