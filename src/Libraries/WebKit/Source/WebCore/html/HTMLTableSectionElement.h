/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 21, 2023.
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

#include "HTMLNames.h"
#include "HTMLTablePartElement.h"

namespace WebCore {

class HTMLTableSectionElement final : public HTMLTablePartElement {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(HTMLTableSectionElement);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(HTMLTableSectionElement);
public:
    static Ref<HTMLTableSectionElement> create(const QualifiedName&, Document&);

    WEBCORE_EXPORT ExceptionOr<Ref<HTMLTableRowElement>> insertRow(int index = -1);
    WEBCORE_EXPORT ExceptionOr<void> deleteRow(int index);

    int numRows() const;

    WEBCORE_EXPORT Ref<HTMLCollection> rows();

private:
    HTMLTableSectionElement(const QualifiedName& tagName, Document&);

    const MutableStyleProperties* additionalPresentationalHintStyle() const final;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_BEGIN(WebCore::HTMLTableSectionElement)
    static bool isType(const WebCore::HTMLElement& element) { return element.hasTagName(WebCore::HTMLNames::theadTag) || element.hasTagName(WebCore::HTMLNames::tfootTag) || element.hasTagName(WebCore::HTMLNames::tbodyTag); }
    static bool isType(const WebCore::Node& node)
    {
        auto* htmlElement = dynamicDowncast<WebCore::HTMLElement>(node);
        return htmlElement && isType(*htmlElement);
    }
SPECIALIZE_TYPE_TRAITS_END()
