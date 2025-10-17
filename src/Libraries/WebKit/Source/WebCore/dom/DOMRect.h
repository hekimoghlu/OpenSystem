/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 14, 2025.
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

#include "DOMRectReadOnly.h"
#include "FloatRect.h"
#include "IntRect.h"

namespace WebCore {

class FloatQuad;

class DOMRect : public DOMRectReadOnly {
public:
    static Ref<DOMRect> create() { return adoptRef(*new DOMRect()); }
    static Ref<DOMRect> create(double x, double y, double width, double height) { return adoptRef(*new DOMRect(x, y, width, height)); }
    static Ref<DOMRect> create(FloatRect rect) { return adoptRef(*new DOMRect(rect.x(), rect.y(), rect.width(), rect.height())); }
    static Ref<DOMRect> create(IntRect rect) { return adoptRef(*new DOMRect(rect.x(), rect.y(), rect.width(), rect.height())); }
    static Ref<DOMRect> fromRect(const DOMRectInit& init) { return create(init.x, init.y, init.width, init.height); }

    void setX(double x) { m_x = x; }
    void setY(double y) { m_y = y; }

    void setWidth(double width) { m_width = width; }
    void setHeight(double height) { m_height = height; }

private:
    DOMRect(double x, double y, double width, double height)
        : DOMRectReadOnly(x, y, width, height)
    {
    }

    DOMRect() = default;
};
static_assert(sizeof(DOMRect) == sizeof(DOMRectReadOnly));

}
