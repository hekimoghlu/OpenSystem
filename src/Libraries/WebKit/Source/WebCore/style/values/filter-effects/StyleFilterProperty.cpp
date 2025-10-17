/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 21, 2024.
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
#include "StyleFilterProperty.h"

#include "CSSFilterProperty.h"
#include "Document.h"
#include "FilterOperations.h"
#include "StyleBlurFunction.h"
#include "StyleBrightnessFunction.h"
#include "StyleContrastFunction.h"
#include "StyleDropShadowFunction.h"
#include "StyleGrayscaleFunction.h"
#include "StyleHueRotateFunction.h"
#include "StyleInvertFunction.h"
#include "StyleOpacityFunction.h"
#include "StyleSaturateFunction.h"
#include "StyleSepiaFunction.h"

namespace WebCore {
namespace Style {

CSS::FilterProperty toCSSFilterProperty(const FilterOperations& filterOperations, const RenderStyle& style)
{
    if (filterOperations.isEmpty())
        return CSS::FilterProperty { CSS::Keyword::None { } };

    CSS::FilterProperty::List list;
    list.value.reserveInitialCapacity(filterOperations.size());

    for (auto& op : filterOperations) {
        switch (op->type()) {
        case FilterOperation::Type::Reference:
            list.value.append(CSS::FilterReference { downcast<ReferenceFilterOperation>(op)->url() });
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
        case FilterOperation::Type::Blur:
            list.value.append(CSS::BlurFunction { toCSSBlur(downcast<BlurFilterOperation>(op), style) });
            break;
        case FilterOperation::Type::DropShadow:
            list.value.append(CSS::DropShadowFunction { toCSSDropShadow(downcast<DropShadowFilterOperation>(op), style) });
            break;
        default:
            ASSERT_NOT_REACHED();
            break;
        }
    }

    return CSS::FilterProperty { WTFMove(list) };
}

static Ref<FilterOperation> createFilterFunctionReference(const String& filterURL, const Document& document)
{
    auto fragment = document.completeURL(filterURL).fragmentIdentifier().toAtomString();
    return ReferenceFilterOperation::create(filterURL, WTFMove(fragment));
}

template<typename T> static Ref<FilterOperation> createFilterPropertyOperation(const T& value, const Document& document, RenderStyle& style, const CSSToLengthConversionData& conversionData)
{
    return WTF::switchOn(value,
        [&](const CSS::FilterReference& reference) {
            return createFilterFunctionReference(reference.url, document);
        },
        [&](const auto& function) {
            return createFilterOperation(function, document, style, conversionData);
        }
    );
}

FilterOperations createFilterOperations(const CSS::FilterProperty& value, const Document& document, RenderStyle& style, const CSSToLengthConversionData& conversionData)
{
    return WTF::switchOn(value,
        [&](CSS::Keyword::None) {
            return FilterOperations { };
        },
        [&](const CSS::FilterProperty::List& list) {
            return FilterOperations { WTF::map(list, [&](const auto& value) {
                return createFilterPropertyOperation(value, document, style, conversionData);
            }) };
        }
    );
}

} // namespace Style
} // namespace WebCore
