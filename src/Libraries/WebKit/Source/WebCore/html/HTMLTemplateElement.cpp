/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 6, 2025.
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
#include "config.h"
#include "HTMLTemplateElement.h"

#include "Document.h"
#include "DocumentFragment.h"
#include "ElementInlines.h"
#include "ElementRareData.h"
#include "HTMLNames.h"
#include "NodeTraversal.h"
#include "ShadowRoot.h"
#include "ShadowRootInit.h"
#include "SlotAssignmentMode.h"
#include "TemplateContentDocumentFragment.h"
#include "markup.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(HTMLTemplateElement);

using namespace HTMLNames;

inline HTMLTemplateElement::HTMLTemplateElement(const QualifiedName& tagName, Document& document)
    : HTMLElement(tagName, document, TypeFlag::HasDidMoveToNewDocument)
{
}

HTMLTemplateElement::~HTMLTemplateElement()
{
    if (m_content)
        m_content->clearHost();
}

Ref<HTMLTemplateElement> HTMLTemplateElement::create(const QualifiedName& tagName, Document& document)
{
    return adoptRef(*new HTMLTemplateElement(tagName, document));
}

DocumentFragment* HTMLTemplateElement::contentIfAvailable() const
{
    return m_content.get();
}

DocumentFragment& HTMLTemplateElement::fragmentForInsertion() const
{
    if (m_declarativeShadowRoot)
        return *m_declarativeShadowRoot;
    return content();
}

DocumentFragment& HTMLTemplateElement::content() const
{
    ASSERT(!m_declarativeShadowRoot);
    if (!m_content)
        m_content = TemplateContentDocumentFragment::create(document().ensureTemplateDocument(), *this);
    return *m_content;
}

const AtomString& HTMLTemplateElement::shadowRootMode() const
{
    static MainThreadNeverDestroyed<const AtomString> open("open"_s);
    static MainThreadNeverDestroyed<const AtomString> closed("closed"_s);

    auto modeString = attributeWithoutSynchronization(HTMLNames::shadowrootmodeAttr);
    if (equalLettersIgnoringASCIICase(modeString, "closed"_s))
        return closed;
    if (equalLettersIgnoringASCIICase(modeString, "open"_s))
        return open;
    return emptyAtom();
}

void HTMLTemplateElement::setShadowRootMode(const AtomString& value)
{
    setAttribute(HTMLNames::shadowrootmodeAttr, value);
}

void HTMLTemplateElement::setDeclarativeShadowRoot(ShadowRoot& shadowRoot)
{
    m_declarativeShadowRoot = shadowRoot;
}

Ref<Node> HTMLTemplateElement::cloneNodeInternal(TreeScope& treeScope, CloningOperation type)
{
    RefPtr<Node> clone;
    switch (type) {
    case CloningOperation::OnlySelf:
        return cloneElementWithoutChildren(treeScope);
    case CloningOperation::SelfWithTemplateContent:
        clone = cloneElementWithoutChildren(treeScope);
        break;
    case CloningOperation::Everything:
        clone = cloneElementWithChildren(treeScope);
        break;
    }
    if (m_content) {
        auto& templateElement = downcast<HTMLTemplateElement>(*clone);
        Ref fragment = templateElement.content();
        content().cloneChildNodes(fragment->document(), fragment);
    }
    return clone.releaseNonNull();
}

void HTMLTemplateElement::didMoveToNewDocument(Document& oldDocument, Document& newDocument)
{
    HTMLElement::didMoveToNewDocument(oldDocument, newDocument);
    if (!m_content)
        return;
    ASSERT_WITH_SECURITY_IMPLICATION(&document() == &newDocument);
    m_content->setTreeScopeRecursively(newDocument.ensureTemplateDocument());
}

void HTMLTemplateElement::attachAsDeclarativeShadowRootIfNeeded(Element& host)
{
    if (m_declarativeShadowRoot) {
        ASSERT(host.shadowRoot());
        return;
    }

    auto modeString = shadowRootMode();
    if (modeString.isEmpty())
        return;

    ASSERT(modeString == "closed"_s || modeString == "open"_s);
    auto mode = modeString == "closed"_s ? ShadowRootMode::Closed : ShadowRootMode::Open;

    auto delegatesFocus = hasAttributeWithoutSynchronization(HTMLNames::shadowrootdelegatesfocusAttr) ? ShadowRootDelegatesFocus::Yes : ShadowRootDelegatesFocus::No;
    auto clonable = hasAttributeWithoutSynchronization(HTMLNames::shadowrootclonableAttr) ? ShadowRootClonable::Yes : ShadowRootClonable::No;
    auto serializable = hasAttributeWithoutSynchronization(HTMLNames::shadowrootserializableAttr) ? ShadowRootSerializable::Yes : ShadowRootSerializable::No;

    auto exceptionOrShadowRoot = host.attachDeclarativeShadow(mode, delegatesFocus, clonable, serializable);
    if (exceptionOrShadowRoot.hasException())
        return;

    auto importedContent = document().importNode(content(), /* deep */ true).releaseReturnValue();
    for (RefPtr<Node> node = NodeTraversal::next(importedContent), next; node; node = next) {
        next = NodeTraversal::next(*node);
        if (auto* templateElement = dynamicDowncast<HTMLTemplateElement>(*node)) {
            if (RefPtr parentElement = node->parentElement())
                templateElement->attachAsDeclarativeShadowRootIfNeeded(*parentElement);
        }
    }

    Ref shadowRoot = exceptionOrShadowRoot.releaseReturnValue();
    shadowRoot->appendChild(WTFMove(importedContent));

    remove();
}

} // namespace WebCore
