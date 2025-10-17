/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 27, 2022.
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

#include "Quad.h"

namespace WebCore {

class CSSQuadValue final : public CSSValue {
public:
    static Ref<CSSQuadValue> create(Quad);

    const Quad& quad() const { return m_quad; }

    String customCSSText() const;
    bool equals(const CSSQuadValue&) const;
    bool canBeCoalesced() const;

private:
    explicit CSSQuadValue(Quad);
    bool m_coalesceIdenticalValues { true };
    Quad m_quad;
};

inline const Quad& CSSValue::quad() const
{
    return downcast<CSSQuadValue>(*this).quad();
}

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_CSS_VALUE(CSSQuadValue, isQuad())
