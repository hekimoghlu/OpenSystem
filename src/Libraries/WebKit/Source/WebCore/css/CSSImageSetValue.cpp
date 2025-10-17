/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 24, 2022.
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
#include "CSSImageSetValue.h"

#include "CSSImageSetOptionValue.h"
#include "CSSImageValue.h"
#include "CSSPrimitiveValue.h"
#include "StyleBuilderState.h"
#include "StyleImageSet.h"
#include <numeric>
#include <wtf/text/StringBuilder.h>

namespace WebCore {

Ref<CSSImageSetValue> CSSImageSetValue::create(CSSValueListBuilder builder)
{
    return adoptRef(*new CSSImageSetValue(WTFMove(builder)));
}

CSSImageSetValue::CSSImageSetValue(CSSValueListBuilder builder)
    : CSSValueContainingVector(ClassType::ImageSet, CommaSeparator, WTFMove(builder))
{
}

String CSSImageSetValue::customCSSText() const
{
    StringBuilder result;
    result.append("image-set("_s);
    for (size_t i = 0; i < this->length(); ++i) {
        if (i > 0)
            result.append(", "_s);
        ASSERT(is<CSSImageSetOptionValue>(item(i)));
        result.append(item(i)->cssText());
    }
    result.append(')');
    return result.toString();
}

RefPtr<StyleImage> CSSImageSetValue::createStyleImage(const Style::BuilderState& state) const
{
    size_t length = this->length();

    Vector<ImageWithScale> images(length, [&](size_t i) {
        auto option = downcast<CSSImageSetOptionValue>(item(i));
        return ImageWithScale { state.createStyleImage(option->image()), option->resolution()->resolveAsResolution<float>(state.cssToLengthConversionData()), option->type() };
    });

    // Sort the images so that they are stored in order from lowest resolution to highest.
    // We want to maintain the authored order for serialization so we create a sorted indexing vector.
    Vector<size_t> sortedIndices(length);
    std::iota(sortedIndices.begin(), sortedIndices.end(), 0);

    std::stable_sort(sortedIndices.begin(), sortedIndices.end(), [&images](size_t lhs, size_t rhs) {
        return images[lhs].scaleFactor < images[rhs].scaleFactor;
    });

    return StyleImageSet::create(WTFMove(images), WTFMove(sortedIndices));
}

} // namespace WebCore
