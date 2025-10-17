/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 25, 2021.
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
#include "StylePaintImage.h"

#include "CSSPaintImageValue.h"
#include "CSSVariableData.h"
#include "CustomPaintImage.h"
#include "PaintWorkletGlobalScope.h"
#include "RenderElement.h"
#include <wtf/PointerComparison.h>

namespace WebCore {

StylePaintImage::StylePaintImage(String&& name, Ref<CSSVariableData>&& arguments)
    : StyleGeneratedImage { Type::PaintImage, StylePaintImage::isFixedSize }
    , m_name { WTFMove(name) }
    , m_arguments { WTFMove(arguments) }
{
}

StylePaintImage::~StylePaintImage() = default;

bool StylePaintImage::operator==(const StyleImage& other) const
{
    // FIXME: Should probably also compare arguments?
    auto* otherPaintImage = dynamicDowncast<StylePaintImage>(other);
    return otherPaintImage && otherPaintImage->m_name == m_name;
}

Ref<CSSValue> StylePaintImage::computedStyleValue(const RenderStyle&) const
{
    return CSSPaintImageValue::create(m_name, m_arguments);
}

bool StylePaintImage::isPending() const
{
    return false;
}

void StylePaintImage::load(CachedResourceLoader&, const ResourceLoaderOptions&)
{
}

RefPtr<Image> StylePaintImage::image(const RenderElement* renderer, const FloatSize& size, bool) const
{
    if (!renderer)
        return &Image::nullImage();

    if (size.isEmpty())
        return nullptr;

    auto* selectedGlobalScope = renderer->document().paintWorkletGlobalScopeForName(m_name);
    if (!selectedGlobalScope)
        return nullptr;

    Locker locker { selectedGlobalScope->paintDefinitionLock() };
    auto* registration = selectedGlobalScope->paintDefinitionMap().get(m_name);

    if (!registration)
        return nullptr;

    // FIXME: Check if argument list matches syntax.
    Vector<String> arguments;
    CSSParserTokenRange localRange(m_arguments->tokenRange());

    while (!localRange.atEnd()) {
        StringBuilder builder;
        while (!localRange.atEnd() && localRange.peek() != CommaToken) {
            if (localRange.peek() == CommentToken)
                localRange.consume();
            else if (localRange.peek().getBlockType() == CSSParserToken::BlockStart) {
                localRange.peek().serialize(builder);
                builder.append(localRange.consumeBlock().serialize(), ')');
            } else
                localRange.consume().serialize(builder);
        }
        if (!localRange.atEnd())
            localRange.consume(); // comma token
        arguments.append(builder.toString());
    }

    return CustomPaintImage::create(*registration, size, *renderer, arguments);
}

bool StylePaintImage::knownToBeOpaque(const RenderElement&) const
{
    return false;
}

FloatSize StylePaintImage::fixedSize(const RenderElement&) const
{
    return { };
}

} // namespace WebCore
