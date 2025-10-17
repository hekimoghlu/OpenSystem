/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 10, 2024.
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

#include "Length.h"
#include "NinePieceImage.h"
#include "RenderStyleConstants.h"
#include <wtf/RefCounted.h>

namespace WebCore {

class StyleReflection : public RefCounted<StyleReflection> {
public:
    static Ref<StyleReflection> create()
    {
        return adoptRef(*new StyleReflection);
    }

    bool operator==(const StyleReflection& o) const
    {
        return m_direction == o.m_direction && m_offset == o.m_offset && m_mask == o.m_mask;
    }

    ReflectionDirection direction() const { return m_direction; }
    const Length& offset() const { return m_offset; }
    const NinePieceImage& mask() const { return m_mask; }

    void setDirection(ReflectionDirection dir) { m_direction = dir; }
    void setOffset(Length offset) { m_offset = WTFMove(offset); }
    void setMask(const NinePieceImage& image) { m_mask = image; }

private:
    StyleReflection()
        : m_offset(0, LengthType::Fixed)
        , m_mask(NinePieceImage::Type::Mask)
    {
        m_mask.setFill(true);
    }
    
    ReflectionDirection m_direction { ReflectionDirection::Below };
    Length m_offset;
    NinePieceImage m_mask;
};

} // namespace WebCore
