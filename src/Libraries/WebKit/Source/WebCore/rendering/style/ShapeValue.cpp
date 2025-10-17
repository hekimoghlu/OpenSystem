/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 15, 2022.
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
#include "ShapeValue.h"

#include "AnimationUtilities.h"
#include "CachedImage.h"
#include "StylePrimitiveNumericTypes+Blending.h"

namespace WebCore {

bool ShapeValue::isImageValid() const
{
    auto image = this->protectedImage();
    if (!image)
        return false;
    if (image->hasCachedImage()) {
        auto* cachedImage = image->cachedImage();
        return cachedImage && cachedImage->hasImage();
    }
    return image->isGeneratedImage();
}

bool ShapeValue::operator==(const ShapeValue& other) const
{
    return std::visit(WTF::makeVisitor(
        []<typename T>(const T& a, const T& b) {
            return a == b;
        },
        [](const auto&, const auto&) {
            return false;
        }
    ), m_value, other.m_value);
}

bool ShapeValue::canBlend(const ShapeValue& other) const
{
    return std::visit(WTF::makeVisitor(
        [](const ShapeAndBox& a, const ShapeAndBox& b) {
            return Style::canBlend(a.shape, b.shape) && a.box == b.box;
        },
        [](const auto&, const auto&) {
            return false;
        }
    ), m_value, other.m_value);
}

Ref<ShapeValue> ShapeValue::blend(const ShapeValue& other, const BlendingContext& context) const
{
    return std::visit(WTF::makeVisitor(
        [&](const ShapeAndBox& a, const ShapeAndBox& b) -> Ref<ShapeValue> {
            return ShapeValue::create(Style::blend(a.shape, b.shape, context), a.box);
        },
        [](const auto&, const auto&) -> Ref<ShapeValue> {
            RELEASE_ASSERT_NOT_REACHED();
        }
    ), m_value, other.m_value);
}

CSSBoxType ShapeValue::effectiveCSSBox() const
{
    return WTF::switchOn(m_value,
        [](const ShapeAndBox& shape) {
            return shape.box == CSSBoxType::BoxMissing ? CSSBoxType::MarginBox : shape.box;
        },
        [](const Ref<StyleImage>&) {
            return CSSBoxType::ContentBox;
        },
        [](const CSSBoxType& box) {
            return box == CSSBoxType::BoxMissing ? CSSBoxType::MarginBox : box;
        }
    );
}

} // namespace WebCore
