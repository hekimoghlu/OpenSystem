/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 2, 2024.
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
#include "FormAssociatedElement.h"

#include "ElementInlines.h"

namespace WebCore {

FormAssociatedElement::FormAssociatedElement(HTMLFormElement* form)
    : m_formSetByParser(form)
{
}

void FormAssociatedElement::setFormInternal(RefPtr<HTMLFormElement>&& newForm)
{
    ASSERT(m_form.get() != newForm);
    m_form = WTFMove(newForm);
}

void FormAssociatedElement::elementInsertedIntoAncestor(Element& element, Node::InsertionType)
{
    ASSERT(&asHTMLElement() == &element);
    if (m_formSetByParser) {
        // The form could have been removed by a script during parsing.
        if (m_formSetByParser->isConnected())
            setForm(m_formSetByParser.get());
        m_formSetByParser = nullptr;
    }

    if (m_form && element.rootElement() != m_form->rootElement())
        setForm(nullptr);
}

void FormAssociatedElement::elementRemovedFromAncestor(Element& element, Node::RemovalType)
{
    ASSERT(&asHTMLElement() == &element);
    // Do not rely on rootNode() because m_form's IsInTreeScope can be outdated.
    if (m_form && &element.traverseToRootNode() != &m_form->traverseToRootNode()) {
        setForm(nullptr);
        resetFormOwner();
    }
}

}
