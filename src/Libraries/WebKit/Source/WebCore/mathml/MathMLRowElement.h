/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 10, 2023.
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

class MathMLRowElement : public MathMLPresentationElement {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(MathMLRowElement);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(MathMLRowElement);
public:
    static Ref<MathMLRowElement> create(const QualifiedName& tagName, Document&);

protected:
    MathMLRowElement(const QualifiedName& tagName, Document&, OptionSet<TypeFlag> = { });
    void childrenChanged(const ChildChange&) override;

    bool acceptsMathVariantAttribute() override;

private:
    RenderPtr<RenderElement> createElementRenderer(RenderStyle&&, const RenderTreePosition&) override;
};

}

#endif // ENABLE(MATHML)
