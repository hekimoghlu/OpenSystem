/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 24, 2022.
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

#include "AccessibilityNodeObject.h"
#include "LayoutRect.h"
#include "PluginViewBase.h"
#include "RenderObject.h"
#include <wtf/Forward.h>
#include <wtf/WeakPtr.h>

namespace WebCore {
    
class AccessibilitySVGRoot;
class AXObjectCache;
class Element;
class HTMLAreaElement;
class HTMLElement;
class HTMLLabelElement;
class HTMLMapElement;
class IntPoint;
class IntSize;
class LocalFrameView;
class Node;
class RenderTextControl;
class RenderView;
class VisibleSelection;

class AccessibilityRenderObject : public AccessibilityNodeObject {
public:
    static Ref<AccessibilityRenderObject> create(AXID, RenderObject&);
    virtual ~AccessibilityRenderObject();
    
    FloatRect frameRect() const final;
    bool isNonLayerSVGObject() const final;

    bool isAttachment() const final;
    bool isDetached() const final { return !m_renderer && AccessibilityNodeObject::isDetached(); }
    bool isOffScreen() const final;
    bool hasBoldFont() const final;
    bool hasItalicFont() const final;
    bool hasPlainText() const final;
    bool hasSameFont(AXCoreObject&) final;
    bool hasSameFontColor(AXCoreObject&) final;
    bool hasSameStyle(AXCoreObject&) final;
    bool hasUnderline() const final;

    void setAccessibleName(const AtomString&) final;

    int layoutCount() const final;

    AccessibilityObject* firstChild() const final;
    AccessibilityObject* lastChild() const final;
    AccessibilityObject* previousSibling() const final;
    AccessibilityObject* nextSibling() const final;
    AccessibilityObject* parentObject() const override;
    AccessibilityObject* observableObject() const override;
    AccessibilityObject* titleUIElement() const override;

    // Should be called on the root accessibility object to kick off a hit test.
    AccessibilityObject* accessibilityHitTest(const IntPoint&) const final;

    Element* anchorElement() const final;
    
    LayoutRect boundingBoxRect() const final;

    RenderObject* renderer() const final { return m_renderer.get(); }
    Document* document() const final;

    URL url() const final;
    CharacterRange selectedTextRange() const final;
    int insertionPointLineNumber() const final;
    String stringValue() const override;
    String helpText() const override;
    String textUnderElement(TextUnderElementMode = TextUnderElementMode()) const override;
    String selectedText() const final;
#if ENABLE(AX_THREAD_TEXT_APIS)
    AXTextRuns textRuns() final;
    AXTextRunLineID listMarkerLineID() const final;
    String listMarkerText() const final;
#endif // ENABLE(AX_THREAD_TEXT_APIS)

    bool isWidget() const final;
    Widget* widget() const final;
    Widget* widgetForAttachmentView() const final;
    AccessibilityChildrenVector documentLinks() final;
    LocalFrameView* documentFrameView() const final;
    bool isPlugin() const final { return is<PluginViewBase>(widget()); }

    void setSelectedTextRange(CharacterRange&&) final;
    bool setValue(const String&) override;

    void addChildren() override;

    IntRect boundsForVisiblePositionRange(const VisiblePositionRange&) const final;
    void setSelectedVisiblePositionRange(const VisiblePositionRange&) const final;
    bool isVisiblePositionRangeInDifferentDocument(const VisiblePositionRange&) const;

    VisiblePosition visiblePositionForIndex(unsigned indexValue, bool lastIndexOK) const final;
    int index(const VisiblePosition&) const final;

    VisiblePosition visiblePositionForIndex(int) const final;
    int indexForVisiblePosition(const VisiblePosition&) const final;

    CharacterRange doAXRangeForLine(unsigned) const final;
    CharacterRange doAXRangeForIndex(unsigned) const final;

    String doAXStringForRange(const CharacterRange&) const final;
    IntRect doAXBoundsForRange(const CharacterRange&) const final;
    IntRect doAXBoundsForRangeUsingCharacterOffset(const CharacterRange&) const final;

    String secureFieldValue() const final;
    void labelText(Vector<AccessibilityText>&) const override;
protected:
    explicit AccessibilityRenderObject(AXID, RenderObject&);
    explicit AccessibilityRenderObject(AXID, Node&);
    void detachRemoteParts(AccessibilityDetachmentType) final;
    ScrollableArea* getScrollableAreaIfScrollable() const final;
    void scrollTo(const IntPoint&) const final;

    bool shouldIgnoreAttributeRole() const override;
    AccessibilityRole determineAccessibilityRole() override;
    bool computeIsIgnored() const override;

#if ENABLE(MATHML)
    virtual bool isIgnoredElementWithinMathTree() const;
#endif

    SingleThreadWeakPtr<RenderObject> m_renderer;

private:
    bool isAccessibilityRenderObject() const final { return true; }
    bool isAllowedChildOfTree() const;
    CharacterRange documentBasedSelectedTextRange() const;
    RefPtr<Element> rootEditableElementForPosition(const Position&) const;
    bool elementIsTextControl(const Element&) const;
    Path elementPath() const final;

    AccessibilityObject* accessibilityImageMapHitTest(HTMLAreaElement&, const IntPoint&) const;
    AccessibilityObject* associatedAXImage(HTMLMapElement&) const;
    AccessibilityObject* elementAccessibilityHitTest(const IntPoint&) const override;

    bool renderObjectIsObservable(RenderObject&) const;
    RenderObject* renderParentObject() const;
    RenderObject* markerRenderer() const;

    bool isSVGImage() const;
    void detachRemoteSVGRoot();
    enum CreationChoice { Create, Retrieve };
    AccessibilitySVGRoot* remoteSVGRootElement(CreationChoice createIfNecessary) const;
    AccessibilityObject* remoteSVGElementHitTest(const IntPoint&) const;
    void offsetBoundingBoxForRemoteSVGElement(LayoutRect&) const;
    bool supportsPath() const final;

#if USE(ATSPI)
    void addNodeOnlyChildren();
    void addCanvasChildren();
#endif // USE(ATSPI)
    void addTextFieldChildren();
    void addImageMapChildren();
    void addAttachmentChildren();
    void addRemoteSVGChildren();
    void addListItemMarker();
#if PLATFORM(COCOA)
    void updateAttachmentViewParents();
#endif
    String expandedTextValue() const override;
    bool supportsExpandedTextValue() const override;
    virtual void updateRoleAfterChildrenCreation();

    bool inheritsPresentationalRole() const override;

    bool shouldGetTextFromNode(const TextUnderElementMode&) const;

#if ENABLE(APPLE_PAY)
    bool isApplePayButton() const;
    ApplePayButtonType applePayButtonType() const;
    String applePayButtonDescription() const;
#endif

    bool canHavePlainText() const;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_ACCESSIBILITY(AccessibilityRenderObject, isAccessibilityRenderObject())
