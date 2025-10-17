/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 2, 2022.
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

#include <wtf/Forward.h>
#include <wtf/OptionSet.h>

namespace WebCore {

class CSSParserTokenRange;
class CSSValue;
struct CSSParserContext;
class CSSValue;

namespace CSSPropertyParserHelpers {

enum class AllowedImageType : uint8_t {
    URLFunction = 1 << 0,
    RawStringAsURL = 1 << 1,
    ImageSet = 1 << 2,
    GeneratedImage = 1 << 3
};

// MARK: <image>
// https://drafts.csswg.org/css-images-4/#image-values

RefPtr<CSSValue> consumeImage(CSSParserTokenRange&, const CSSParserContext&, OptionSet<AllowedImageType> = { AllowedImageType::URLFunction, AllowedImageType::ImageSet, AllowedImageType::GeneratedImage });
RefPtr<CSSValue> consumeImageOrNone(CSSParserTokenRange&, const CSSParserContext&, OptionSet<AllowedImageType> = { AllowedImageType::URLFunction, AllowedImageType::ImageSet, AllowedImageType::GeneratedImage });

} // namespace CSSPropertyParserHelpers
} // namespace WebCore
