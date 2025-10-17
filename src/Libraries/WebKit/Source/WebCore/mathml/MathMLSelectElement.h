/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 11, 2022.
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

#if ENABLE(MATHML)

#include "MathMLRowElement.h"

namespace WebCore {

class MathMLSelectElement final : public MathMLRowElement {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(MathMLSelectElement);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(MathMLSelectElement);
public:
    static Ref<MathMLSelectElement> create(const QualifiedName& tagName, Document&);
    static bool isMathMLEncoding(const AtomString& value);
    static bool isSVGEncoding(const AtomString& value);
    static bool isHTMLEncoding(const AtomString& value);

private:
    MathMLSelectElement(const QualifiedName& tagName, Document&);
    RenderPtr<RenderElement> createElementRenderer(RenderStyle&&, const RenderTreePosition&) final;

    bool childShouldCreateRenderer(const Node&) const final;

    void finishParsingChildren() final;
    void childrenChanged(const ChildChange&) final;
    void attributeChanged(const QualifiedName&, const AtomString& oldValue, const AtomString& newValue, AttributeModificationReason = AttributeModificationReason::Directly) final;
    void defaultEventHandler(Event&) final;
    bool willRespondToMouseClickEventsWithEditability(Editability) const final;

    void toggle();
    int getSelectedActionChildAndIndex(Element*& selectedChild);
    Element* getSelectedActionChild();
    Element* getSelectedSemanticsChild();

    void updateSelectedChild() final;
    RefPtr<Element> m_selectedChild;
};

}

#endif // ENABLE(MATHML)
