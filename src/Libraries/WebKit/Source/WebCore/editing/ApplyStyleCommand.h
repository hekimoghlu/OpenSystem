/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 17, 2023.
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

#include "CompositeEditCommand.h"
#include "HTMLElement.h"
#include "WritingDirection.h"

namespace WebCore {

class CSSPrimitiveValue;
class EditingStyle;
class StyleChange;

enum ShouldIncludeTypingStyle {
    IncludeTypingStyle,
    IgnoreTypingStyle
};

class ApplyStyleCommand : public CompositeEditCommand {
public:
    enum class InlineStyleRemovalMode : uint8_t { IfNeeded, Always, None };
    enum class AddStyledElement : bool { No, Yes };
    typedef bool (*IsInlineElementToRemoveFunction)(const Element*);

    static Ref<ApplyStyleCommand> create(Ref<Document>&& document, const EditingStyle* style, EditAction action = EditAction::ChangeAttributes, ApplyStylePropertyLevel level = ApplyStylePropertyLevel::Default)
    {
        return adoptRef(*new ApplyStyleCommand(WTFMove(document), style, action, level));
    }
    static Ref<ApplyStyleCommand> create(Ref<Document>&& document, const EditingStyle* style, const Position& start, const Position& end, EditAction action = EditAction::ChangeAttributes, ApplyStylePropertyLevel level = ApplyStylePropertyLevel::Default)
    {
        return adoptRef(*new ApplyStyleCommand(WTFMove(document), style, start, end, action, level));
    }
    static Ref<ApplyStyleCommand> create(Ref<Element>&& element, bool removeOnly = false, EditAction action = EditAction::ChangeAttributes)
    {
        return adoptRef(*new ApplyStyleCommand(WTFMove(element), removeOnly, action));
    }
    static Ref<ApplyStyleCommand> create(Ref<Document>&& document, const EditingStyle* style, IsInlineElementToRemoveFunction isInlineElementToRemoveFunction, EditAction action = EditAction::ChangeAttributes)
    {
        return adoptRef(*new ApplyStyleCommand(WTFMove(document), style, isInlineElementToRemoveFunction, action));
    }

private:
    ApplyStyleCommand(Ref<Document>&&, const EditingStyle*, EditAction, ApplyStylePropertyLevel);
    ApplyStyleCommand(Ref<Document>&&, const EditingStyle*, const Position& start, const Position& end, EditAction, ApplyStylePropertyLevel);
    ApplyStyleCommand(Ref<Element>&&, bool removeOnly, EditAction);
    ApplyStyleCommand(Ref<Document>&&, const EditingStyle*, bool (*isInlineElementToRemove)(const Element*), EditAction);

    void doApply() override;
    bool shouldDispatchInputEvents() const final { return false; }

    // style-removal helpers
    bool isStyledInlineElementToRemove(Element*) const;
    bool shouldApplyInlineStyleToRun(EditingStyle&, Node* runStart, Node* pastEndNode);
    void removeConflictingInlineStyleFromRun(EditingStyle&, RefPtr<Node>& runStart, RefPtr<Node>& runEnd, Node* pastEndNode);
    bool removeInlineStyleFromElement(EditingStyle&, HTMLElement&, InlineStyleRemovalMode = InlineStyleRemovalMode::IfNeeded, EditingStyle* extractedStyle = nullptr);
    inline bool shouldRemoveInlineStyleFromElement(EditingStyle& style, HTMLElement& element) { return removeInlineStyleFromElement(style, element, InlineStyleRemovalMode::None); }
    void replaceWithSpanOrRemoveIfWithoutAttributes(HTMLElement&);
    bool removeImplicitlyStyledElement(EditingStyle&, HTMLElement&, InlineStyleRemovalMode, EditingStyle* extractedStyle);
    bool removeCSSStyle(EditingStyle&, HTMLElement&, InlineStyleRemovalMode = InlineStyleRemovalMode::IfNeeded, EditingStyle* extractedStyle = nullptr);
    RefPtr<HTMLElement> highestAncestorWithConflictingInlineStyle(EditingStyle&, Node*);
    void applyInlineStyleToPushDown(Node&, EditingStyle*);
    void pushDownInlineStyleAroundNode(EditingStyle&, Node*);
    void removeInlineStyle(EditingStyle&, const Position& start, const Position& end);
    bool nodeFullySelected(Element&, const Position& start, const Position& end) const;

    // style-application helpers
    void applyBlockStyle(EditingStyle&);
    void applyRelativeFontStyleChange(EditingStyle*);
    void applyInlineStyle(EditingStyle&);
    void fixRangeAndApplyInlineStyle(EditingStyle&, const Position& start, const Position& end);
    void applyInlineStyleToNodeRange(EditingStyle&, Node& startNode, Node* pastEndNode);
    void addBlockStyle(const StyleChange&, HTMLElement&);
    void addInlineStyleIfNeeded(EditingStyle*, Node& start, Node& end, AddStyledElement = AddStyledElement::Yes);
    Position positionToComputeInlineStyleChange(Node&, RefPtr<Node>& dummyElement);
    void applyInlineStyleChange(Node& startNode, Node& endNode, StyleChange&, AddStyledElement);
    void splitTextAtStart(const Position& start, const Position& end);
    void splitTextAtEnd(const Position& start, const Position& end);
    void splitTextElementAtStart(const Position& start, const Position& end);
    void splitTextElementAtEnd(const Position& start, const Position& end);
    bool shouldSplitTextElement(Element*, EditingStyle&);
    bool isValidCaretPositionInTextNode(const Position& position);
    bool mergeStartWithPreviousIfIdentical(const Position& start, const Position& end);
    bool mergeEndWithNextIfIdentical(const Position& start, const Position& end);
    void cleanupUnstyledAppleStyleSpans(ContainerNode* dummySpanAncestor);

    bool surroundNodeRangeWithElement(Node& start, Node& end, Ref<Element>&&);
    float computedFontSize(Node*);
    void joinChildTextNodes(Node*, const Position& start, const Position& end);

    RefPtr<HTMLElement> splitAncestorsWithUnicodeBidi(Node*, bool before, WritingDirection allowedDirection);
    void removeEmbeddingUpToEnclosingBlock(Node* node, Node* unsplitAncestor);

    void updateStartEnd(const Position& newStart, const Position& newEnd);
    Position startPosition();
    Position endPosition();

    RefPtr<EditingStyle> m_style;
    ApplyStylePropertyLevel m_propertyLevel { ApplyStylePropertyLevel::Default };
    Position m_start;
    Position m_end;
    bool m_useEndingSelection;
    RefPtr<Element> m_styledInlineElement;
    bool m_removeOnly;
    IsInlineElementToRemoveFunction m_isInlineElementToRemoveFunction { nullptr };
};

enum ShouldStyleAttributeBeEmpty { AllowNonEmptyStyleAttribute, StyleAttributeShouldBeEmpty };
bool isEmptyFontTag(const Element*, ShouldStyleAttributeBeEmpty = StyleAttributeShouldBeEmpty);
bool isLegacyAppleStyleSpan(const Node*);
bool isStyleSpanOrSpanWithOnlyStyleAttribute(const Element&);
Ref<HTMLElement> createStyleSpanElement(Document&);

} // namespace WebCore
