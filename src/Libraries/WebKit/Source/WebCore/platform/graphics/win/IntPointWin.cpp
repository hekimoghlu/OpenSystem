/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 28, 2025.
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
#include "IntPoint.h"

#include <windows.h>
#include <wtf/MathExtras.h>

namespace WebCore {

IntPoint::IntPoint(const POINT& p)
    : m_x(p.x)
    , m_y(p.y)
{
}

IntPoint::operator POINT() const
{
    POINT p = {m_x, m_y};
    return p;
}

IntPoint::IntPoint(const POINTS& p)
    : m_x(p.x)
    , m_y(p.y)
{
}

IntPoint::operator POINTS() const
{
    POINTS p = { static_cast<SHORT>(m_x), static_cast<SHORT>(m_y) };
    return p;
}

}
