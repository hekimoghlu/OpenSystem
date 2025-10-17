/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 20, 2025.
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

class MathMLPaddedElement final : public MathMLRowElement {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(MathMLPaddedElement);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(MathMLPaddedElement);
public:
    static Ref<MathMLPaddedElement> create(const QualifiedName& tagName, Document&);
    // FIXME: Pseudo-units are not supported yet (https://bugs.webkit.org/show_bug.cgi?id=85730).
    const Length& width();
    const Length& height();
    const Length& depth();
    const Length& lspace();
    const Length& voffset();
private:
    MathMLPaddedElement(const QualifiedName& tagName, Document&);
    RenderPtr<RenderElement> createElementRenderer(RenderStyle&&, const RenderTreePosition&) final;
    void attributeChanged(const QualifiedName&, const AtomString& oldValue, const AtomString& newValue, AttributeModificationReason) final;

    std::optional<Length> m_width;
    std::optional<Length> m_height;
    std::optional<Length> m_depth;
    std::optional<Length> m_lspace;
    std::optional<Length> m_voffset;
};

}

#endif // ENABLE(MATHML)
