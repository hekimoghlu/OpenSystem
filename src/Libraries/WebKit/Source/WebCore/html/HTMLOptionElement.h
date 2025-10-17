/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 15, 2025.
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

#include "HTMLElement.h"

namespace WebCore {

class HTMLSelectElement;

enum class AllowStyleInvalidation : bool { No, Yes };

class HTMLOptionElement final : public HTMLElement {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(HTMLOptionElement);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(HTMLOptionElement);
public:
    static Ref<HTMLOptionElement> create(Document&);
    static Ref<HTMLOptionElement> create(const QualifiedName&, Document&);
    static ExceptionOr<Ref<HTMLOptionElement>> createForLegacyFactoryFunction(Document&, String&& text, const AtomString& value, bool defaultSelected, bool selected);

    WEBCORE_EXPORT String text() const;
    void setText(String&&);

    WEBCORE_EXPORT HTMLFormElement* form() const;

    WEBCORE_EXPORT int index() const;

    WEBCORE_EXPORT String value() const;
    WEBCORE_EXPORT void setValue(const AtomString&);

    WEBCORE_EXPORT bool selected(AllowStyleInvalidation = AllowStyleInvalidation::Yes) const;
    WEBCORE_EXPORT void setSelected(bool);

    WEBCORE_EXPORT HTMLSelectElement* ownerSelectElement() const;

    WEBCORE_EXPORT String label() const;
    WEBCORE_EXPORT String displayLabel() const;
    WEBCORE_EXPORT void setLabel(const AtomString&);

    bool ownElementDisabled() const { return m_disabled; }

    WEBCORE_EXPORT bool isDisabledFormControl() const final;

    String textIndentedToRespectGroupLabel() const;

    void setSelectedState(bool, AllowStyleInvalidation = AllowStyleInvalidation::Yes);
    bool selectedWithoutUpdate() const { return m_isSelected; }

private:
    HTMLOptionElement(const QualifiedName&, Document&);

    bool isFocusable() const final;
    bool rendererIsNeeded(const RenderStyle&) final { return false; }
    bool matchesDefaultPseudoClass() const final;

    void attributeChanged(const QualifiedName&, const AtomString& oldValue, const AtomString& newValue, AttributeModificationReason) final;

    bool accessKeyAction(bool) final;

    void childrenChanged(const ChildChange&) final;

    void willResetComputedStyle() final;

    String collectOptionInnerText() const;

    bool m_disabled { false };
    bool m_isSelected { false };
    bool m_isDefault { false };
};

} // namespace
