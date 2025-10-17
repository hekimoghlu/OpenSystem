/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 15, 2023.
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

#include "HTMLStackItem.h"
#include <wtf/Forward.h>
#include <wtf/Noncopyable.h>
#include <wtf/Ref.h>
#include <wtf/TZoneMalloc.h>

namespace WebCore {

class ContainerNode;
class Element;
class QualifiedName;

// NOTE: The HTML5 spec uses a backwards (grows downward) stack.  We're using
// more standard (grows upwards) stack terminology here.
class HTMLElementStack {
    WTF_MAKE_TZONE_ALLOCATED(HTMLElementStack);
    WTF_MAKE_NONCOPYABLE(HTMLElementStack);
public:
    HTMLElementStack() = default;
    ~HTMLElementStack();

    class ElementRecord {
        WTF_MAKE_TZONE_ALLOCATED(ElementRecord);
        WTF_MAKE_NONCOPYABLE(ElementRecord);
    public:
        ElementRecord(HTMLStackItem&&, std::unique_ptr<ElementRecord>);
        ~ElementRecord();

        Element& element() const { return m_item.element(); }
        Ref<Element> protectedElement() const { return m_item.element(); }
        ContainerNode& node() const { return m_item.node(); }
        ElementName elementName() const { return m_item.elementName(); }
        HTMLStackItem& stackItem() { return m_item; }
        const HTMLStackItem& stackItem() const { return m_item; }

        void replaceElement(HTMLStackItem&&);

        bool isAbove(ElementRecord&) const;

        ElementRecord* next() const { return m_next.get(); }

    private:
        friend class HTMLElementStack;

        std::unique_ptr<ElementRecord> releaseNext() { return WTFMove(m_next); }
        void setNext(std::unique_ptr<ElementRecord> next) { m_next = WTFMove(next); }

        HTMLStackItem m_item;
        std::unique_ptr<ElementRecord> m_next;
    };

    unsigned stackDepth() const { return m_stackDepth; }

    // Inlining this function is a (small) performance win on the parsing
    // benchmark.
    Element& top() const { return m_top->element(); }
    ContainerNode& topNode() const { return m_top->node(); }
    ElementName topElementName() const { return m_top->elementName(); }
    HTMLStackItem& topStackItem() const { return m_top->stackItem(); }

    HTMLStackItem* oneBelowTop() const;
    ElementRecord& topRecord() const;
    ElementRecord* find(Element&) const;
    ElementRecord* furthestBlockForFormattingElement(Element&) const;
    ElementRecord* topmost(ElementName) const;

    void insertAbove(HTMLStackItem&&, ElementRecord&);

    void push(HTMLStackItem&&);
    void pushRootNode(HTMLStackItem&&);
    void pushHTMLHtmlElement(HTMLStackItem&&);
    void pushHTMLHeadElement(HTMLStackItem&&);
    void pushHTMLBodyElement(HTMLStackItem&&);

    void pop();
    void popUntil(ElementName);
    void popUntil(Element&);
    void popUntilPopped(ElementName);
    void popUntilPopped(Element&);
    void popUntilNumberedHeaderElementPopped();
    void popUntilTableScopeMarker(); // "clear the stack back to a table context" in the spec.
    void popUntilTableBodyScopeMarker(); // "clear the stack back to a table body context" in the spec.
    void popUntilTableRowScopeMarker(); // "clear the stack back to a table row context" in the spec.
    void popUntilForeignContentScopeMarker();
    void popHTMLHeadElement();
    void popHTMLBodyElement();
    void popAll();

    static bool isMathMLTextIntegrationPoint(HTMLStackItem&);
    static bool isHTMLIntegrationPoint(HTMLStackItem&);

    void remove(Element&);
    void removeHTMLHeadElement(Element&);

    bool contains(Element&) const;

    bool inScope(Element&) const;
    bool inScope(ElementName) const;
    bool inListItemScope(ElementName) const;
    bool inTableScope(ElementName) const;
    bool inButtonScope(ElementName) const;
    bool inSelectScope(ElementName) const;

    bool hasNumberedHeaderElementInScope() const;

    bool hasOnlyOneElement() const;
    bool secondElementIsHTMLBodyElement() const;
    bool hasTemplateInHTMLScope() const;
    Element& htmlElement() const;
    Element& headElement() const;
    Element& bodyElement() const;

    ContainerNode& rootNode() const;

#if ENABLE(TREE_DEBUGGING)
    void show();
#endif

private:
    void pushCommon(HTMLStackItem&&);
    void pushRootNodeCommon(HTMLStackItem&&);
    void popCommon();
    void removeNonTopCommon(Element&);

    std::unique_ptr<ElementRecord> m_top;

    // We remember the root node, <head> and <body> as they are pushed. Their
    // ElementRecords keep them alive. The root node is never popped.
    // FIXME: We don't currently require type-specific information about
    // these elements so we haven't yet bothered to plumb the types all the
    // way down through createElement, etc.
    CheckedPtr<ContainerNode> m_rootNode;
    CheckedPtr<Element> m_headElement;
    CheckedPtr<Element> m_bodyElement;
    unsigned m_stackDepth { 0 };
};
    
} // namespace WebCore
