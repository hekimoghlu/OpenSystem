/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 29, 2022.
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
#include "config.h"
#include "CSSQuadValue.h"
#include "CSSValue.h"

namespace WebCore {

CSSQuadValue::CSSQuadValue(Quad quad)
    : CSSValue(ClassType::Quad)
    , m_coalesceIdenticalValues(true)
    , m_quad(WTFMove(quad))
{
}

Ref<CSSQuadValue> CSSQuadValue::create(Quad quad)
{
    return adoptRef(*new CSSQuadValue(WTFMove(quad)));
}

String CSSQuadValue::customCSSText() const
{
    return m_quad.cssText();
}

bool CSSQuadValue::equals(const CSSQuadValue& other) const
{
    return m_quad.equals(other.m_quad);
}

bool CSSQuadValue::canBeCoalesced() const
{
    Ref top = m_quad.top();
    Ref right = m_quad.right();
    Ref left = m_quad.left();
    Ref bottom = m_quad.bottom();
    return m_coalesceIdenticalValues && top->equals(right) && top->equals(left) && top->equals(bottom);
}

} // namespace WebCore
