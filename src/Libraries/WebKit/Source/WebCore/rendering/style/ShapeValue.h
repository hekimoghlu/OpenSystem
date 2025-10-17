/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 4, 2024.
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

#include "RenderStyleConstants.h"
#include "StyleBasicShape.h"
#include "StyleImage.h"

namespace WebCore {

struct BlendingContext;

class ShapeValue : public RefCounted<ShapeValue> {
public:
    static Ref<ShapeValue> create(Style::BasicShape&& shape, CSSBoxType cssBox)
    {
        return adoptRef(*new ShapeValue(WTFMove(shape), cssBox));
    }

    static Ref<ShapeValue> create(CSSBoxType boxShape)
    {
        return adoptRef(*new ShapeValue(boxShape));
    }

    static Ref<ShapeValue> create(Ref<StyleImage>&& image)
    {
        return adoptRef(*new ShapeValue(WTFMove(image)));
    }

    enum class Type { Shape, Box, Image };
    Type type() const;

    const Style::BasicShape* shape() const;
    CSSBoxType cssBox() const;
    CSSBoxType effectiveCSSBox() const;
    StyleImage* image() const;
    RefPtr<StyleImage> protectedImage() const;
    bool isImageValid() const;

    void setImage(Ref<StyleImage>&& image)
    {
        ASSERT(type() == Type::Image);
        m_value = WTFMove(image);
    }

    bool canBlend(const ShapeValue&) const;
    Ref<ShapeValue> blend(const ShapeValue&, const BlendingContext&) const;

    bool operator==(const ShapeValue&) const;

private:
    struct ShapeAndBox {
        Style::BasicShape shape;
        CSSBoxType box;

        bool operator==(const ShapeAndBox&) const = default;
    };

    ShapeValue(Style::BasicShape&& shape, CSSBoxType cssBox)
        : m_value(ShapeAndBox { WTFMove(shape), cssBox })
    {
    }

    explicit ShapeValue(Ref<StyleImage>&& image)
        : m_value(WTFMove(image))
    {
    }

    explicit ShapeValue(CSSBoxType cssBox)
        : m_value(cssBox)
    {
    }

    std::variant<ShapeAndBox, Ref<StyleImage>, CSSBoxType> m_value;
};

inline ShapeValue::Type ShapeValue::type() const
{
    return WTF::switchOn(m_value,
        [](const ShapeAndBox&) { return Type::Shape; },
        [](const Ref<StyleImage>&) { return Type::Image; },
        [](const CSSBoxType&) { return Type::Box; }
    );
}

inline const Style::BasicShape* ShapeValue::shape() const
{
    return WTF::switchOn(m_value,
        [](const ShapeAndBox& shape) -> const Style::BasicShape* { return &shape.shape; },
        [](const Ref<StyleImage>&) -> const Style::BasicShape* { return nullptr; },
        [](const CSSBoxType&) -> const Style::BasicShape* { return nullptr; }
    );
}

inline CSSBoxType ShapeValue::cssBox() const
{
    return WTF::switchOn(m_value,
        [](const ShapeAndBox& shape) { return shape.box; },
        [](const Ref<StyleImage>&) { return CSSBoxType::BoxMissing; },
        [](const CSSBoxType& box) { return box; }
    );
}

inline StyleImage* ShapeValue::image() const
{
    return WTF::switchOn(m_value,
        [](const ShapeAndBox&) -> StyleImage* { return nullptr; },
        [](const Ref<StyleImage>& image) -> StyleImage* { return image.ptr(); },
        [](const CSSBoxType&) -> StyleImage* { return nullptr; }
    );
}

inline RefPtr<StyleImage> ShapeValue::protectedImage() const
{
    return WTF::switchOn(m_value,
        [](const ShapeAndBox&) -> RefPtr<StyleImage> { return nullptr; },
        [](const Ref<StyleImage>& image) -> RefPtr<StyleImage> { return image.ptr(); },
        [](const CSSBoxType&) -> RefPtr<StyleImage> { return nullptr; }
    );
}

} // namespace WebCore
