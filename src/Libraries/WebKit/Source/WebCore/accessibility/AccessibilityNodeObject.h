/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 25, 2022.
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

#include "AccessibilityObject.h"
#include "LayoutRect.h"
#include <wtf/Forward.h>

namespace WebCore {

class AXObjectCache;
class Element;
class HTMLLabelElement;
class Node;

class AccessibilityNodeObject : public AccessibilityObject {
public:
    static Ref<AccessibilityNodeObject> create(AXID, Node&);
    virtual ~AccessibilityNodeObject();

    void init() override;

    bool canvasHasFallbackContent() const final;

    bool isBusy() const final;
    bool isDetached() const override { return !m_node; }
    bool isRadioInput() const final;
    bool isFieldset() const final;
    bool isInputImage() const final;
    bool isMultiSelectable() const override;
    bool isNativeImage() const;
    bool isNativeTextControl() const final;
    bool isSecureField() const final;
    bool isSearchField() const final;

    bool isChecked() const final;
    bool isEnabled() const override;
    bool isIndeterminate() const override;
    bool isPressed() const final;
    bool isRequired() const final;
    bool supportsARIAOwns() const final;

    bool supportsDropping() const final;
    bool supportsDragging() const final;
    bool isGrabbed() final;
    Vector<String> determineDropEffects() const final;

    bool canSetSelectedAttribute() const override;

    Node* node() const final { return m_node.get(); }
    Document* document() const override;
    LocalFrameView* documentFrameView() const override;

    void setFocused(bool) override;
    bool isFocused() const final;
    bool canSetFocusAttribute() const override;
    unsigned headingLevel() const final;

    bool canSetValueAttribute() const override;

    String valueDescription() const override;
    float valueForRange() const override;
    float maxValueForRange() const override;
    float minValueForRange() const override;
    float stepValueForRange() const override;

    AccessibilityOrientation orientation() const override;

    AccessibilityButtonState checkboxOrRadioValue() const final;

    URL url() const override;
    unsigned hierarchicalLevel() const final;
    String textUnderElement(TextUnderElementMode = TextUnderElementMode()) const override;
    String accessibilityDescriptionForChildren() const;
    String description() const override;
    String helpText() const override;
    String title() const override;
    String text() const final;
    void alternativeText(Vector<AccessibilityText>&) const;
    void helpText(Vector<AccessibilityText>&) const;
    String stringValue() const override;
    WallTime dateTimeValue() const final;
    SRGBA<uint8_t> colorValue() const final;
    String ariaLabeledByAttribute() const final;
    bool hasAccNameAttribute() const;
    bool hasAttributesRequiredForInclusion() const final;
    bool hasClickHandler() const final;
    void setIsExpanded(bool) final;

    Element* actionElement() const override;
    Element* anchorElement() const override;
    RefPtr<Element> popoverTargetElement() const final;
    AccessibilityObject* internalLinkElement() const final;
    AccessibilityChildrenVector radioButtonGroup() const final;
   
    virtual void changeValueByPercent(float percentChange);
 
    AccessibilityObject* firstChild() const override;
    AccessibilityObject* lastChild() const override;
    AccessibilityObject* previousSibling() const override;
    AccessibilityObject* nextSibling() const override;
    AccessibilityObject* parentObject() const override;

    bool matchesTextAreaRole() const;

    void increment() override;
    void decrement() override;
    bool toggleDetailsAncestor() final;

    LayoutRect elementRect() const override;

#if ENABLE(AX_THREAD_TEXT_APIS)
    TextEmissionBehavior emitTextAfterBehavior() const final;
#endif

protected:
    explicit AccessibilityNodeObject(AXID, Node*);
    void detachRemoteParts(AccessibilityDetachmentType) override;

    AccessibilityRole m_ariaRole { AccessibilityRole::Unknown };
#ifndef NDEBUG
    bool m_initialized { false };
#endif

    AccessibilityRole determineAccessibilityRole() override;
    enum class TreatStyleFormatGroupAsInline : bool { No, Yes };
    AccessibilityRole determineAccessibilityRoleFromNode(TreatStyleFormatGroupAsInline = TreatStyleFormatGroupAsInline::No) const;
    AccessibilityRole roleFromInputElement(const HTMLInputElement&) const;
    AccessibilityRole ariaRoleAttribute() const final { return m_ariaRole; }
    virtual AccessibilityRole determineAriaRoleAttribute() const;
    AccessibilityRole remapAriaRoleDueToParent(AccessibilityRole) const;

    bool computeIsIgnored() const override;
    void addChildren() override;
    void clearChildren() override;
    void updateChildrenIfNecessary() override;
    bool canHaveChildren() const override;
    AccessibilityChildrenVector visibleChildren() override;
    bool isDescendantOfBarrenParent() const final;
    void updateOwnedChildren();
    AccessibilityObject* ownerParentObject() const;
    
    enum class StepAction : bool { Decrement, Increment };
    void alterRangeValue(StepAction);
    void changeValueByStep(StepAction);
    // This returns true if it's focusable but it's not content editable and it's not a control or ARIA control.
    bool isGenericFocusableElement() const;

    VisiblePositionRange visiblePositionRange() const final;
    VisiblePositionRange selectedVisiblePositionRange() const final;
    VisiblePositionRange visiblePositionRangeForLine(unsigned) const final;
    VisiblePosition visiblePositionForIndex(int) const override;
    int indexForVisiblePosition(const VisiblePosition&) const override;

    bool elementAttributeValue(const QualifiedName&) const;

    const String liveRegionStatus() const final;
    const String liveRegionRelevant() const final;
    bool liveRegionAtomic() const final;

    String accessKey() const final;
    bool isLabelable() const;
    AccessibilityObject* controlForLabelElement() const final;
    String textAsLabelFor(const AccessibilityObject&) const;
    String textForLabelElements(Vector<Ref<HTMLElement>>&&) const;
    HTMLLabelElement* labelElementContainer() const;

    String ariaAccessibilityDescription() const;
    Vector<Ref<Element>> ariaLabeledByElements() const;
    String descriptionForElements(const Vector<Ref<Element>>&) const;
    LayoutRect boundingBoxRect() const override;
    String ariaDescribedByAttribute() const final;

    AccessibilityObject* captionForFigure() const;
    virtual void labelText(Vector<AccessibilityText>&) const;
private:
    bool isAccessibilityNodeObject() const final { return true; }
    void accessibilityText(Vector<AccessibilityText>&) const override;
    void visibleText(Vector<AccessibilityText>&) const;
    String alternativeTextForWebArea() const;
    void ariaLabeledByText(Vector<AccessibilityText>&) const;
    bool usesAltTagForTextComputation() const;
    bool roleIgnoresTitle() const;
    bool postKeyboardKeysForValueChange(StepAction);
    void setNodeValue(StepAction, float);
    bool performDismissAction() final;
    bool hasTextAlternative() const;
    LayoutRect checkboxOrRadioRect() const;

    void setNeedsToUpdateChildren() override { m_childrenDirty = true; }
    bool needsToUpdateChildren() const final { return m_childrenDirty; }
    void setNeedsToUpdateSubtree() final { m_subtreeDirty = true; }

    bool isDescendantOfElementType(const HashSet<QualifiedName>&) const;
protected:
    WeakPtr<Node, WeakPtrImplWithEventTargetData> m_node;
};

namespace Accessibility {

RefPtr<HTMLElement> controlForLabelElement(const HTMLLabelElement&);
Vector<Ref<HTMLElement>> labelsForElement(Element*);

} // namespace Accessibility

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_BEGIN(WebCore::AccessibilityNodeObject) \
    static bool isType(const WebCore::AccessibilityObject& object) { return object.isAccessibilityNodeObject(); } \
SPECIALIZE_TYPE_TRAITS_END()
