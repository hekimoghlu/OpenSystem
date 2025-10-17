/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 26, 2023.
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

#include "HTMLTablePartElement.h"

namespace WebCore {

class HTMLTableCellElement final : public HTMLTablePartElement {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(HTMLTableCellElement);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(HTMLTableCellElement);
public:
    // These limits are defined in the HTML specification:
    // - https://html.spec.whatwg.org/#dom-tdth-colspan
    // - https://html.spec.whatwg.org/#dom-tdth-rowspan
    static constexpr unsigned minColspan = 1;
    static constexpr unsigned maxColspan = 1000;
    static constexpr unsigned defaultColspan = 1;
    static constexpr unsigned minRowspan = 0;
    static constexpr unsigned maxRowspan = 65534;
    static constexpr unsigned defaultRowspan = 1;

    static Ref<HTMLTableCellElement> create(const QualifiedName&, Document&);

    WEBCORE_EXPORT int cellIndex() const;
    WEBCORE_EXPORT unsigned colSpan() const;
    unsigned rowSpan() const;
    WEBCORE_EXPORT unsigned rowSpanForBindings() const;

    void setCellIndex(int);
    WEBCORE_EXPORT void setColSpan(unsigned);
    WEBCORE_EXPORT void setRowSpanForBindings(unsigned);

    String abbr() const;
    String axis() const;
    String headers() const;
    WEBCORE_EXPORT const AtomString& scope() const;
    WEBCORE_EXPORT void setScope(const AtomString&);

    WEBCORE_EXPORT HTMLTableCellElement* cellAbove() const;

private:
    HTMLTableCellElement(const QualifiedName&, Document&);

    void attributeChanged(const QualifiedName&, const AtomString& oldValue, const AtomString& newValue, AttributeModificationReason) override;
    bool hasPresentationalHintsForAttribute(const QualifiedName&) const override;
    void collectPresentationalHintsForAttribute(const QualifiedName&, const AtomString&, MutableStyleProperties&) override;
    const MutableStyleProperties* additionalPresentationalHintStyle() const override;

    bool isURLAttribute(const Attribute&) const override;

    void addSubresourceAttributeURLs(ListHashSet<URL>&) const override;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_BEGIN(WebCore::HTMLTableCellElement)
    static bool isType(const WebCore::HTMLElement& element) { return element.hasTagName(WebCore::HTMLNames::tdTag) || element.hasTagName(WebCore::HTMLNames::thTag); }
    static bool isType(const WebCore::Node& node)
    {
        auto* htmlElement = dynamicDowncast<WebCore::HTMLElement>(node);
        return htmlElement && isType(*htmlElement);
    }
SPECIALIZE_TYPE_TRAITS_END()
