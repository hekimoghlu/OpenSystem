/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 23, 2023.
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

#include "MathMLElement.h"
#include <wtf/RefCounted.h>

namespace WebCore {

class MathMLStyle: public RefCounted<MathMLStyle> {
public:
    static Ref<MathMLStyle> create();

    MathMLElement::MathVariant mathVariant() const { return m_mathVariant; }
    void setMathVariant(MathMLElement::MathVariant mathvariant) { m_mathVariant = mathvariant; }

    void resolveMathMLStyle(RenderObject*);
    static void resolveMathMLStyleTree(RenderObject*);

private:
    const MathMLStyle* getMathMLStyle(RenderObject* renderer);
    RenderObject* getMathMLParentNode(RenderObject*);
    void updateStyleIfNeeded(RenderObject*, MathMLElement::MathVariant);

    MathMLElement::MathVariant m_mathVariant { MathMLElement::MathVariant::None };
};

}

#endif // ENABLE(MATHML)
