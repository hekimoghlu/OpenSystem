/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 28, 2025.
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
#include "HitTestSource.h"
#include <memory>
#include <wtf/Forward.h>
#include <wtf/UniqueRef.h>
#include <wtf/Vector.h>
#include <wtf/text/AtomString.h>

namespace JSC {
class JSValue;
}

namespace WebCore {

class CSSStyleSheet;
class CSSStyleSheetObservableArray;
class ContainerNode;
class CustomElementRegistry;
class Document;
class Element;
class FloatPoint;
class JSDOMGlobalObject;
class HTMLAnchorElement;
class HTMLImageElement;
class HTMLLabelElement;
class HTMLMapElement;
class LayoutPoint;
class LegacyRenderSVGResourceContainer;
class IdTargetObserverRegistry;
class Node;
class QualifiedName;
class RadioButtonGroups;
class SVGElement;
class ShadowRoot;
class TreeScopeOrderedMap;
class WeakPtrImplWithEventTargetData;
struct SVGResourcesMap;

class TreeScope {
    friend class Document;

public:
    TreeScope* parentTreeScope() const { return m_parentTreeScope; }
    void setParentTreeScope(TreeScope&);

    WEBCORE_EXPORT void ref() const;
    WEBCORE_EXPORT void deref() const;

    Element* focusedElementInScope();
    Element* pointerLockElement() const;

    void setCustomElementRegistry(Ref<CustomElementRegistry>&&);
    CustomElementRegistry* customElementRegistry() const { return m_customElementRegistry.get(); }
    WEBCORE_EXPORT ExceptionOr<Ref<Element>> createElementForBindings(const AtomString& tagName);
    WEBCORE_EXPORT ExceptionOr<Ref<Element>> createElementNS(const AtomString& namespaceURI, const AtomString& qualifiedName);
    WEBCORE_EXPORT Ref<Element> createElement(const QualifiedName&, bool createdByParser);
    WEBCORE_EXPORT ExceptionOr<Ref<Node>> importNode(Node& nodeToImport, bool deep);

    WEBCORE_EXPORT RefPtr<Element> getElementById(const AtomString&) const;
    WEBCORE_EXPORT RefPtr<Element> getElementById(const String&) const;
    RefPtr<Element> getElementById(StringView) const;
    const Vector<WeakRef<Element, WeakPtrImplWithEventTargetData>>* getAllElementsById(const AtomString&) const;
    inline bool hasElementWithId(const AtomString&) const; // Defined in TreeScopeInlines.h.
    inline bool containsMultipleElementsWithId(const AtomString& id) const; // Defined in TreeScopeInlines.h.
    void addElementById(const AtomString& elementId, Element&, bool notifyObservers = true);
    void removeElementById(const AtomString& elementId, Element&, bool notifyObservers = true);

    WEBCORE_EXPORT RefPtr<Element> getElementByName(const AtomString&) const;
    inline bool hasElementWithName(const AtomString&) const; // Defined in TreeScopeInlines.h.
    inline bool containsMultipleElementsWithName(const AtomString&) const; // Defined in TreeScopeInlines.h.
    void addElementByName(const AtomString&, Element&);
    void removeElementByName(const AtomString&, Element&);

    Document& documentScope() const { return m_documentScope.get(); }
    Ref<Document> protectedDocumentScope() const;
    static constexpr ptrdiff_t documentScopeMemoryOffset() { return OBJECT_OFFSETOF(TreeScope, m_documentScope); }

    // https://dom.spec.whatwg.org/#retarget
    Ref<Node> retargetToScope(Node&) const;

    WEBCORE_EXPORT Node* ancestorNodeInThisScope(Node*) const;
    WEBCORE_EXPORT Element* ancestorElementInThisScope(Element*) const;

    void addImageMap(HTMLMapElement&);
    void removeImageMap(HTMLMapElement&);
    RefPtr<HTMLMapElement> getImageMap(const AtomString&) const;

    void addImageElementByUsemap(const AtomString&, HTMLImageElement&);
    void removeImageElementByUsemap(const AtomString&, HTMLImageElement&);
    RefPtr<HTMLImageElement> imageElementByUsemap(const AtomString&) const;

    // For accessibility.
    bool shouldCacheLabelsByForAttribute() const { return !!m_labelsByForAttribute; }
    void addLabel(const AtomString& forAttributeValue, HTMLLabelElement&);
    void removeLabel(const AtomString& forAttributeValue, HTMLLabelElement&);
    const Vector<WeakRef<Element, WeakPtrImplWithEventTargetData>>* labelElementsForId(const AtomString& forAttributeValue);

    WEBCORE_EXPORT RefPtr<Element> elementFromPoint(double clientX, double clientY, HitTestSource = HitTestSource::Script);
    WEBCORE_EXPORT Vector<RefPtr<Element>> elementsFromPoint(double clientX, double clientY, HitTestSource = HitTestSource::Script);

    // Find first anchor with the given name.
    // First searches for an element with the given ID, but if that fails, then looks
    // for an anchor with the given name. ID matching is always case sensitive, but
    // Anchor name matching is case sensitive in strict mode and not case sensitive in
    // quirks mode for historical compatibility reasons.
    RefPtr<Element> findAnchor(StringView name);
    bool isMatchingAnchor(HTMLAnchorElement&, StringView name);

    inline ContainerNode& rootNode() const; // Defined in ContainerNode.h
    Ref<ContainerNode> protectedRootNode() const;

    inline IdTargetObserverRegistry& idTargetObserverRegistry();
    IdTargetObserverRegistry* idTargetObserverRegistryIfExists() { return m_idTargetObserverRegistry.get(); }

    RadioButtonGroups& radioButtonGroups();

    JSC::JSValue adoptedStyleSheetWrapper(JSDOMGlobalObject&);
    std::span<const Ref<CSSStyleSheet>> adoptedStyleSheets() const;
    ExceptionOr<void> setAdoptedStyleSheets(Vector<Ref<CSSStyleSheet>>&&);

    void addSVGResource(const AtomString& id, LegacyRenderSVGResourceContainer&);
    void removeSVGResource(const AtomString& id);
    LegacyRenderSVGResourceContainer* lookupLegacySVGResoureById(const AtomString& id) const;

    void addPendingSVGResource(const AtomString& id, SVGElement&);
    bool isIdOfPendingSVGResource(const AtomString& id) const;
    bool isPendingSVGResource(SVGElement&, const AtomString& id) const;
    void clearHasPendingSVGResourcesIfPossible(SVGElement&);
    void removeElementFromPendingSVGResources(SVGElement&);
    WeakHashSet<SVGElement, WeakPtrImplWithEventTargetData> removePendingSVGResource(const AtomString&);
    void markPendingSVGResourcesForRemoval(const AtomString&);
    RefPtr<SVGElement> takeElementFromPendingSVGResourcesForRemovalMap(const AtomString&);

protected:
    TreeScope(ShadowRoot&, Document&, RefPtr<CustomElementRegistry>&&);
    explicit TreeScope(Document&);
    ~TreeScope();

    void destroyTreeScopeData();
    void setDocumentScope(Document& document)
    {
        m_documentScope = document;
    }

    RefPtr<Node> nodeFromPoint(const LayoutPoint& clientPoint, LayoutPoint* localPoint, HitTestSource);

private:
    IdTargetObserverRegistry& ensureIdTargetObserverRegistry();
    CSSStyleSheetObservableArray& ensureAdoptedStyleSheets();

    SVGResourcesMap& svgResourcesMap() const;
    bool isElementWithPendingSVGResources(SVGElement&) const;

    CheckedRef<ContainerNode> m_rootNode;
    std::reference_wrapper<Document> m_documentScope;
    TreeScope* m_parentTreeScope;

    RefPtr<CustomElementRegistry> m_customElementRegistry;

    std::unique_ptr<TreeScopeOrderedMap> m_elementsById;
    std::unique_ptr<TreeScopeOrderedMap> m_elementsByName;
    std::unique_ptr<TreeScopeOrderedMap> m_imageMapsByName;
    std::unique_ptr<TreeScopeOrderedMap> m_imagesByUsemap;
    std::unique_ptr<TreeScopeOrderedMap> m_labelsByForAttribute;

    std::unique_ptr<IdTargetObserverRegistry> m_idTargetObserverRegistry;

    std::unique_ptr<RadioButtonGroups> m_radioButtonGroups;
    RefPtr<CSSStyleSheetObservableArray> m_adoptedStyleSheets;

    std::unique_ptr<SVGResourcesMap> m_svgResourcesMap;
};

TreeScope* commonTreeScope(Node*, Node*);

} // namespace WebCore
