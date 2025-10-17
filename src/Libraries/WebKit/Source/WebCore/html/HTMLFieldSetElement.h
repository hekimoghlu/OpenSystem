/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 27, 2023.
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

#include "HTMLFormControlElement.h"
#include <wtf/WeakHashSet.h>

namespace WebCore {

class HTMLFieldSetElement final : public HTMLFormControlElement {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(HTMLFieldSetElement);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(HTMLFieldSetElement);
public:
    static Ref<HTMLFieldSetElement> create(const QualifiedName&, Document&, HTMLFormElement*);

    HTMLLegendElement* legend() const;

    Ref<HTMLCollection> elements();

    void addInvalidDescendant(const HTMLElement&);
    void removeInvalidDescendant(const HTMLElement&);

private:
    HTMLFieldSetElement(const QualifiedName&, Document&, HTMLFormElement*);
    ~HTMLFieldSetElement();

    bool isDisabledFormControl() const final;
    bool isActuallyDisabled() const final;
    bool isEnumeratable() const final { return true; }
    bool supportsFocus() const final;
    RenderPtr<RenderElement> createElementRenderer(RenderStyle&&, const RenderTreePosition&) final;
    const AtomString& formControlType() const final;
    bool computeWillValidate() const final { return false; }
    void attributeChanged(const QualifiedName&, const AtomString& oldValue, const AtomString& newValue, AttributeModificationReason) final;
    void disabledStateChanged() final;
    void childrenChanged(const ChildChange&) final;
    void didMoveToNewDocument(Document& oldDocument, Document& newDocument) final;

    bool matchesValidPseudoClass() const final;
    bool matchesInvalidPseudoClass() const final;

    WeakHashSet<HTMLElement, WeakPtrImplWithEventTargetData> m_invalidDescendants;
};

} // namespace
