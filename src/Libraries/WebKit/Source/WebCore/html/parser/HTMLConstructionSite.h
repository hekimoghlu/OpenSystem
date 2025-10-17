/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 27, 2023.
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
#include "HTMLElementStack.h"
#include "HTMLFormattingElementList.h"
#include "ParserContentPolicy.h"
#include <wtf/CheckedRef.h>
#include <wtf/FixedVector.h>
#include <wtf/Noncopyable.h>
#include <wtf/RefPtr.h>
#include <wtf/SetForScope.h>
#include <wtf/Vector.h>

namespace WebCore {

struct AtomStringWithCode {
    AtomString string;
    uint64_t code { 0 };
};

}

namespace WTF {
template<> struct VectorTraits<WebCore::AtomStringWithCode> : SimpleClassVectorTraits { };
}

namespace WebCore {

struct HTMLConstructionSiteTask {
    enum Operation {
        Insert,
        InsertAlreadyParsedChild,
        Reparent,
        TakeAllChildrenAndReparent,
    };

    explicit HTMLConstructionSiteTask(Operation op)
        : operation(op)
        , selfClosing(false)
    {
    }

    ContainerNode* oldParent()
    {
        // It's sort of ugly, but we store the |oldParent| in the |child| field
        // of the task so that we don't bloat the HTMLConstructionSiteTask
        // object in the common case of the Insert operation.
        return downcast<ContainerNode>(child.get());
    }

    Ref<ContainerNode> protectedNonNullParent() const { return *parent; }
    Ref<Node> protectedNonNullChild() const { return *child; }
    Ref<Node> protectedNonNullNextChild() const { return *nextChild; }

    Operation operation;
    RefPtr<ContainerNode> parent;
    RefPtr<Node> nextChild;
    RefPtr<Node> child;
    bool selfClosing;
};

} // namespace WebCore

namespace WTF {
template<> struct VectorTraits<WebCore::HTMLConstructionSiteTask> : SimpleClassVectorTraits { };
} // namespace WTF

namespace WebCore {

enum WhitespaceMode {
    AllWhitespace,
    NotAllWhitespace,
    WhitespaceUnknown
};

class AtomHTMLToken;
class CustomElementRegistry;
class Document;
class Element;
class HTMLFormElement;
class HTMLTemplateElement;
class JSCustomElementInterface;
struct CustomElementConstructionData;

class HTMLConstructionSite {
    WTF_MAKE_NONCOPYABLE(HTMLConstructionSite);
public:
    HTMLConstructionSite(Document&, OptionSet<ParserContentPolicy>, unsigned maximumDOMTreeDepth);
    HTMLConstructionSite(DocumentFragment&, OptionSet<ParserContentPolicy>, unsigned maximumDOMTreeDepth, CustomElementRegistry*);
    ~HTMLConstructionSite();

    void executeQueuedTasks();

    void setDefaultCompatibilityMode();
    void finishedParsing();

    void insertDoctype(AtomHTMLToken&&);
    void insertComment(AtomHTMLToken&&);
    void insertCommentOnDocument(AtomHTMLToken&&);
    void insertCommentOnHTMLHtmlElement(AtomHTMLToken&&);
    void insertHTMLElement(AtomHTMLToken&&);
    void insertHTMLTemplateElement(AtomHTMLToken&&);
    std::unique_ptr<CustomElementConstructionData> insertHTMLElementOrFindCustomElementInterface(AtomHTMLToken&&);
    void insertCustomElement(Ref<Element>&&, Vector<Attribute>&&);
    void insertSelfClosingHTMLElement(AtomHTMLToken&&);
    void insertFormattingElement(AtomHTMLToken&&);
    void insertHTMLHeadElement(AtomHTMLToken&&);
    void insertHTMLBodyElement(AtomHTMLToken&&);
    void insertHTMLFormElement(AtomHTMLToken&&);
    void insertScriptElement(AtomHTMLToken&&);
    void insertTextNode(const String&);
    void insertForeignElement(AtomHTMLToken&&, const AtomString& namespaceURI);

    void insertHTMLHtmlStartTagBeforeHTML(AtomHTMLToken&&);
    void insertHTMLHtmlStartTagInBody(AtomHTMLToken&&);
    void insertHTMLBodyStartTagInBody(AtomHTMLToken&&);

    void reparent(HTMLElementStack::ElementRecord& newParent, HTMLElementStack::ElementRecord& child);
    // insertAlreadyParsedChild assumes that |child| has already been parsed (i.e., we're just
    // moving it around in the tree rather than parsing it for the first time). That means
    // this function doesn't call beginParsingChildren / finishParsingChildren.
    void insertAlreadyParsedChild(HTMLStackItem& newParent, HTMLElementStack::ElementRecord& child);
    void takeAllChildrenAndReparent(HTMLStackItem& newParent, HTMLElementStack::ElementRecord& oldParent);

    HTMLStackItem createElementFromSavedToken(const HTMLStackItem&);

    bool shouldFosterParent() const;
    void fosterParent(Ref<Node>&&);

    std::optional<unsigned> indexOfFirstUnopenFormattingElement() const;
    void reconstructTheActiveFormattingElements();

    void generateImpliedEndTags();
    void generateImpliedEndTagsWithExclusion(ElementName);
    void generateImpliedEndTagsWithExclusion(const AtomString& tagName);

    bool inQuirksMode() { return m_inQuirksMode; }

    bool isEmpty() const { return !m_openElements.stackDepth(); }
    Element& currentElement() const { return m_openElements.top(); }
    ContainerNode& currentNode() const { return m_openElements.topNode(); }
    Ref<ContainerNode> protectedCurrentNode() const { return m_openElements.topNode(); }
    ElementName currentElementName() const { return m_openElements.topElementName(); }
    HTMLStackItem& currentStackItem() const { return m_openElements.topStackItem(); }
    HTMLStackItem* oneBelowTop() const { return m_openElements.oneBelowTop(); }
    TreeScope& treeScopeForCurrentNode();
    Document& ownerDocumentForCurrentNode();
    Ref<Document> protectedOwnerDocumentForCurrentNode() { return ownerDocumentForCurrentNode(); }
    HTMLElementStack& openElements() const { return m_openElements; }
    HTMLFormattingElementList& activeFormattingElements() const { return m_activeFormattingElements; }
    bool currentIsRootNode() { return &m_openElements.topNode() == &m_openElements.rootNode(); }

    Element& head() const { return m_head.element(); }
    HTMLStackItem& headStackItem() { return m_head; }

    void setForm(HTMLFormElement*);
    HTMLFormElement* form() const { return m_form.get(); }
    RefPtr<HTMLFormElement> takeForm();

    OptionSet<ParserContentPolicy> parserContentPolicy() { return m_parserContentPolicy; }

#if ENABLE(TELEPHONE_NUMBER_DETECTION)
    bool isTelephoneNumberParsingEnabled() { return document().isTelephoneNumberParsingEnabled(); }
#endif

    class RedirectToFosterParentGuard {
        WTF_MAKE_NONCOPYABLE(RedirectToFosterParentGuard);
    public:
        explicit RedirectToFosterParentGuard(HTMLConstructionSite& tree)
            : m_redirectAttachToFosterParentChange(tree.m_redirectAttachToFosterParent, true)
        { }

    private:
        SetForScope<bool> m_redirectAttachToFosterParentChange;
    };

    static bool isFormattingTag(TagName);

private:
    Document& document() const { return m_document.get(); }

    // In the common case, this queue will have only one task because most
    // tokens produce only one DOM mutation.
    typedef Vector<HTMLConstructionSiteTask, 1> TaskQueue;

    void setCompatibilityMode(DocumentCompatibilityMode);
    void setCompatibilityModeFromDoctype(const AtomString& name, const String& publicId, const String& systemId);

    void attachLater(Ref<ContainerNode>&& parent, Ref<Node>&& child, bool selfClosing = false);

    void findFosterSite(HTMLConstructionSiteTask&);

    std::tuple<RefPtr<HTMLElement>, RefPtr<JSCustomElementInterface>, RefPtr<CustomElementRegistry>> createHTMLElementOrFindCustomElementInterface(AtomHTMLToken&);
    Ref<HTMLElement> createHTMLElement(AtomHTMLToken&);
    Ref<Element> createElement(AtomHTMLToken&, const AtomString& namespaceURI);

    void mergeAttributesFromTokenIntoElement(AtomHTMLToken&&, Element&);
    void dispatchDocumentElementAvailableIfNeeded();

    Ref<Document> protectedDocument() const;
    Ref<ContainerNode> protectedAttachmentRoot() const;

    // m_head has to be destroyed after destroying CheckedRef of m_document and m_attachmentRoot
    HTMLStackItem m_head;

    WeakRef<Document, WeakPtrImplWithEventTargetData> m_document;
    
    // This is the root ContainerNode to which the parser attaches all newly
    // constructed nodes. It points to a DocumentFragment when parsing fragments
    // and a Document in all other cases.
    WeakRef<ContainerNode, WeakPtrImplWithEventTargetData> m_attachmentRoot;
    
    RefPtr<HTMLFormElement> m_form;
    mutable HTMLElementStack m_openElements;
    mutable HTMLFormattingElementList m_activeFormattingElements;

    TaskQueue m_taskQueue;

    OptionSet<ParserContentPolicy> m_parserContentPolicy;
    RefPtr<CustomElementRegistry> m_registry;
    bool m_isParsingFragment;

    // http://www.whatwg.org/specs/web-apps/current-work/multipage/tokenization.html#parsing-main-intable
    // In the "in table" insertion mode, we sometimes get into a state where
    // "whenever a node would be inserted into the current node, it must instead
    // be foster parented."  This flag tracks whether we're in that state.
    bool m_redirectAttachToFosterParent;

    unsigned m_maximumDOMTreeDepth;

    bool m_inQuirksMode;
};

} // namespace WebCore
