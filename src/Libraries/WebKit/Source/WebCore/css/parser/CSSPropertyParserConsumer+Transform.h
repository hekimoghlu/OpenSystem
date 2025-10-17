/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 15, 2024.
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

#include <optional>
#include <wtf/Forward.h>

namespace WebCore {

class CSSParserTokenRange;
class CSSToLengthConversionData;
class CSSValue;
class TransformOperations;

struct CSSParserContext;

namespace CSSPropertyParserHelpers {

// MARK: <transform> consuming (CSSValue)
RefPtr<CSSValue> consumeTransform(CSSParserTokenRange&, const CSSParserContext&);
RefPtr<CSSValue> consumeTransformList(CSSParserTokenRange&, const CSSParserContext&);
RefPtr<CSSValue> consumeTransformFunction(CSSParserTokenRange&, const CSSParserContext&);
RefPtr<CSSValue> consumeTranslate(CSSParserTokenRange&, const CSSParserContext&);
RefPtr<CSSValue> consumeScale(CSSParserTokenRange&, const CSSParserContext&);
RefPtr<CSSValue> consumeRotate(CSSParserTokenRange&, const CSSParserContext&);

// MARK: <transform> parsing (raw)
std::optional<TransformOperations> parseTransformRaw(const String&, const CSSParserContext&, const CSSToLengthConversionData&);

} // namespace CSSPropertyParserHelpers
} // namespace WebCore
