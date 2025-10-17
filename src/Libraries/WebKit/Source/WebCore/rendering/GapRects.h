/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 3, 2021.
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

#include "LayoutRect.h"

namespace WebCore {

    struct GapRects {
        const LayoutRect& left() const { return m_left; }
        const LayoutRect& center() const { return m_center; }
        const LayoutRect& right() const { return m_right; }
        
        void uniteLeft(const LayoutRect& r) { m_left.unite(r); }
        void uniteCenter(const LayoutRect& r) { m_center.unite(r); }
        void uniteRight(const LayoutRect& r) { m_right.unite(r); }
        void unite(const GapRects& o) { uniteLeft(o.left()); uniteCenter(o.center()); uniteRight(o.right()); }

        operator LayoutRect() const
        {
            LayoutRect result = m_left;
            result.unite(m_center);
            result.unite(m_right);
            return result;
        }

        friend bool operator==(const GapRects&, const GapRects&) = default;

    private:
        LayoutRect m_left;
        LayoutRect m_center;
        LayoutRect m_right;
    };

} // namespace WebCore
