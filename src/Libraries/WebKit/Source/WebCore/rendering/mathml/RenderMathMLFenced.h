/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 9, 2023.
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

#include "RenderMathMLFencedOperator.h"
#include "RenderMathMLRow.h"

namespace WebCore {

class MathMLRowElement;

class RenderMathMLFenced final : public RenderMathMLRow {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(RenderMathMLFenced);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(RenderMathMLFenced);
public:
    RenderMathMLFenced(MathMLRowElement&, RenderStyle&&);
    virtual ~RenderMathMLFenced();

    StringImpl* separators() const { return m_separators.get(); }
    String openingBrace() const { return m_open; }
    String closingBrace() const { return m_close; }

    RenderMathMLFencedOperator* closeFenceRenderer() const { return m_closeFenceRenderer.get(); }
    void setCloseFenceRenderer(RenderMathMLFencedOperator& renderer) { m_closeFenceRenderer = renderer; }

    void updateFromElement();

private:
    ASCIILiteral renderName() const final { return "RenderMathMLFenced"_s; }

    String m_open;
    String m_close;
    RefPtr<StringImpl> m_separators;

    SingleThreadWeakPtr<RenderMathMLFencedOperator> m_closeFenceRenderer;
};

}

SPECIALIZE_TYPE_TRAITS_RENDER_OBJECT(RenderMathMLFenced, isRenderMathMLFenced())

#endif // ENABLE(MATHML)
