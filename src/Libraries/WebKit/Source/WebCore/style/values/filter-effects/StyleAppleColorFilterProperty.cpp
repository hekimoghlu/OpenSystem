/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 5, 2023.
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
#include "StyleAppleColorFilterProperty.h"

#include "CSSAppleColorFilterProperty.h"
#include "FilterOperations.h"
#include "StyleAppleInvertLightnessFunction.h"
#include "StyleBrightnessFunction.h"
#include "StyleContrastFunction.h"
#include "StyleGrayscaleFunction.h"
#include "StyleHueRotateFunction.h"
#include "StyleInvertFunction.h"
#include "StyleOpacityFunction.h"
#include "StyleSaturateFunction.h"
#include "StyleSepiaFunction.h"

namespace WebCore {
namespace Style {

CSS::AppleColorFilterProperty toCSSAppleColorFilterProperty(const FilterOperations& filterOperations, const RenderStyle& style)
{
    if (filterOperations.isEmpty())
        return CSS::AppleColorFilterProperty { CSS::Keyword::None { } };

    CSS::AppleColorFilterProperty::List list;
    list.value.reserveInitialCapacity(filterOperations.size());

    for (auto& op : filterOperations) {
        switch (op->type()) {
        case FilterOperation::Type::AppleInvertLightness:
            list.value.append(CSS::AppleInvertLightnessFunction { toCSSAppleInvertLightness(downcast<InvertLightnessFilterOperation>(op), style) });
            break;
        case FilterOperation::Type::Grayscale:
            list.value.append(CSS::GrayscaleFunction { toCSSGrayscale(downcast<BasicColorMatrixFilterOperation>(op), style) });
            break;
        case FilterOperation::Type::Sepia:
            list.value.append(CSS::SepiaFunction { toCSSSepia(downcast<BasicColorMatrixFilterOperation>(op), style) });
            break;
        case FilterOperation::Type::Saturate:
            list.value.append(CSS::SaturateFunction { toCSSSaturate(downcast<BasicColorMatrixFilterOperation>(op), style) });
            break;
        case FilterOperation::Type::HueRotate:
            list.value.append(CSS::HueRotateFunction { toCSSHueRotate(downcast<BasicColorMatrixFilterOperation>(op), style) });
            break;
        case FilterOperation::Type::Invert:
            list.value.append(CSS::InvertFunction { toCSSInvert(downcast<BasicComponentTransferFilterOperation>(op), style) });
            break;
        case FilterOperation::Type::Opacity:
            list.value.append(CSS::OpacityFunction { toCSSOpacity(downcast<BasicComponentTransferFilterOperation>(op), style) });
            break;
        case FilterOperation::Type::Brightness:
            list.value.append(CSS::BrightnessFunction { toCSSBrightness(downcast<BasicComponentTransferFilterOperation>(op), style) });
            break;
        case FilterOperation::Type::Contrast:
            list.value.append(CSS::ContrastFunction { toCSSContrast(downcast<BasicComponentTransferFilterOperation>(op), style) });
            break;
        default:
            ASSERT_NOT_REACHED();
            break;
        }
    }

    return CSS::AppleColorFilterProperty { WTFMove(list) };
}

template<typename T> static Ref<FilterOperation> createAppleColorFilterPropertyOperation(const T& value, const Document& document, RenderStyle& style, const CSSToLengthConversionData& conversionData)
{
    return WTF::switchOn(value,
        [&](const auto& function) {
            return createFilterOperation(function, document, style, conversionData);
        }
    );
}

FilterOperations createAppleColorFilterOperations(const CSS::AppleColorFilterProperty& value, const Document& document, RenderStyle& style, const CSSToLengthConversionData& conversionData)
{
    return WTF::switchOn(value,
        [&](CSS::Keyword::None) {
            return FilterOperations { };
        },
        [&](const CSS::AppleColorFilterProperty::List& list) {
            return FilterOperations { WTF::map(list, [&](const auto& value) {
                return createAppleColorFilterPropertyOperation(value, document, style, conversionData);
            }) };
        }
    );
}

} // namespace Style
} // namespace WebCore
