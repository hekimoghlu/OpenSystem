/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 26, 2021.
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

#include "MathMLPresentationElement.h"

namespace WebCore {

class MathMLTokenElement : public MathMLPresentationElement {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(MathMLTokenElement);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(MathMLTokenElement);
public:
    static Ref<MathMLTokenElement> create(const QualifiedName& tagName, Document&);

    static std::optional<char32_t> convertToSingleCodePoint(StringView);

protected:
    MathMLTokenElement(const QualifiedName& tagName, Document&);
    void childrenChanged(const ChildChange&) override;

private:
    RenderPtr<RenderElement> createElementRenderer(RenderStyle&&, const RenderTreePosition&) override;
    bool childShouldCreateRenderer(const Node&) const final;

    void didAttachRenderers() final;

    bool isMathMLToken() const final { return true; }
    bool acceptsMathVariantAttribute() final { return true; }
};

}

#endif // ENABLE(MATHML)
