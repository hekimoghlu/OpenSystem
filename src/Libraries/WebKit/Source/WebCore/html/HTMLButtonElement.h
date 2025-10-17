/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 10, 2024.
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

namespace WebCore {

class RenderButton;

class HTMLButtonElement final : public HTMLFormControlElement {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(HTMLButtonElement);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(HTMLButtonElement);
public:
    static Ref<HTMLButtonElement> create(const QualifiedName&, Document&, HTMLFormElement*);
    static Ref<HTMLButtonElement> create(Document&);

    WEBCORE_EXPORT void setType(const AtomString&);
    
    const AtomString& value() const;

    bool willRespondToMouseClickEventsWithEditability(Editability) const final;

    RenderButton* renderer() const;

    bool isExplicitlySetSubmitButton() const;

    bool isDevolvableWidget() const override { return true; }

private:
    HTMLButtonElement(const QualifiedName& tagName, Document&, HTMLFormElement*);

    enum Type { SUBMIT, RESET, BUTTON };

    const AtomString& formControlType() const final;

    RenderPtr<RenderElement> createElementRenderer(RenderStyle&&, const RenderTreePosition&) final;

    int defaultTabIndex() const final;

    void attributeChanged(const QualifiedName&, const AtomString& oldValue, const AtomString& newValue, AttributeModificationReason) final;
    bool hasPresentationalHintsForAttribute(const QualifiedName&) const final;
    void defaultEventHandler(Event&) final;

    bool appendFormData(DOMFormData&) final;

    bool isEnumeratable() const final { return true; }
    bool isLabelable() const final { return true; }
    bool isInteractiveContent() const final { return true; }

    bool isSuccessfulSubmitButton() const final;
    bool matchesDefaultPseudoClass() const final;
    bool isActivatedSubmit() const final;
    void setActivatedSubmit(bool flag) final;

    bool isURLAttribute(const Attribute&) const final;

    bool canStartSelection() const final { return false; }

    bool isOptionalFormControl() const final { return true; }
    bool computeWillValidate() const final;

    bool isSubmitButton() const final;

    Type m_type;
    bool m_isActivatedSubmit;
};

} // namespace
