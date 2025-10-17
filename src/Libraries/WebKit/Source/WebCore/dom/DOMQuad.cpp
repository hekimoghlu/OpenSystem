/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 13, 2024.
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
#include "DOMQuad.h"

#include "DOMPoint.h"
#include "DOMRect.h"
#include <wtf/MathExtras.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {
using namespace WTF;

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(DOMQuad);

DOMQuad::DOMQuad(const DOMPointInit& p1, const DOMPointInit& p2, const DOMPointInit& p3, const DOMPointInit& p4)
    : m_p1(DOMPoint::create(p1))
    , m_p2(DOMPoint::create(p2))
    , m_p3(DOMPoint::create(p3))
    , m_p4(DOMPoint::create(p4))
{
}

//  p1------p2
//  |       |
//  |       |
//  p4------p3
DOMQuad::DOMQuad(const DOMRectInit& r)
    : m_p1(DOMPoint::create(r.x, r.y))
    , m_p2(DOMPoint::create(r.x + r.width, r.y))
    , m_p3(DOMPoint::create(r.x + r.width, r.y + r.height))
    , m_p4(DOMPoint::create(r.x, r.y + r.height))
{
}

Ref<DOMRect> DOMQuad::getBounds() const
{
    double left = nanPropagatingMin(nanPropagatingMin(nanPropagatingMin(m_p1->x(), m_p2->x()), m_p3->x()), m_p4->x());
    double top = nanPropagatingMin(nanPropagatingMin(nanPropagatingMin(m_p1->y(), m_p2->y()), m_p3->y()), m_p4->y());
    double right = nanPropagatingMax(nanPropagatingMax(nanPropagatingMax(m_p1->x(), m_p2->x()), m_p3->x()), m_p4->x());
    double bottom = nanPropagatingMax(nanPropagatingMax(nanPropagatingMax(m_p1->y(), m_p2->y()), m_p3->y()), m_p4->y());

    return DOMRect::create(left, top, right - left, bottom - top);
}

}
