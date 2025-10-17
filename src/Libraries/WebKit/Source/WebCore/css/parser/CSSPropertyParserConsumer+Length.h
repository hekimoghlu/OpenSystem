/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 28, 2024.
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

#include "CSSPropertyParserOptions.h"
#include "Length.h"
#include <wtf/Forward.h>

namespace WebCore {

class CSSParserTokenRange;
class CSSPrimitiveValue;
struct CSSParserContext;

namespace CSSPropertyParserHelpers {

// MARK: - Consumer functions

RefPtr<CSSPrimitiveValue> consumeLength(CSSParserTokenRange&, const CSSParserContext&, ValueRange = ValueRange::All, UnitlessQuirk = UnitlessQuirk::Forbid);
RefPtr<CSSPrimitiveValue> consumeLength(CSSParserTokenRange&, const CSSParserContext&, CSSParserMode overrideParserMode, ValueRange = ValueRange::All, UnitlessQuirk = UnitlessQuirk::Forbid);

} // namespace CSSPropertyParserHelpers
} // namespace WebCore
