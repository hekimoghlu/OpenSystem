/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 28, 2025.
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

class HTMLTableColElement final : public HTMLTablePartElement {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(HTMLTableColElement);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(HTMLTableColElement);
public:
    static Ref<HTMLTableColElement> create(const QualifiedName& tagName, Document&);

    unsigned span() const { return m_span; }
    WEBCORE_EXPORT void setSpan(unsigned);

    String width() const;

private:
    HTMLTableColElement(const QualifiedName& tagName, Document&);

    void attributeChanged(const QualifiedName&, const AtomString& oldValue, const AtomString& newValue, AttributeModificationReason) final;
    bool hasPresentationalHintsForAttribute(const QualifiedName&) const final;
    void collectPresentationalHintsForAttribute(const QualifiedName&, const AtomString&, MutableStyleProperties&) final;
    const MutableStyleProperties* additionalPresentationalHintStyle() const final;

    unsigned m_span;
};

} //namespace
