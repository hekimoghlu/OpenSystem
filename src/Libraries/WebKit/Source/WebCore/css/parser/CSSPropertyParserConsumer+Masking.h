/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 14, 2024.
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

namespace WebCore {

class CSSParserTokenRange;
class CSSValue;
struct CSSParserContext;

namespace CSSPropertyParserHelpers {

// <'clip'> = <rect()> | auto
// https://drafts.fxtf.org/css-masking/#propdef-clip
RefPtr<CSSValue> consumeClip(CSSParserTokenRange&, const CSSParserContext&);

// <'clip-path'> = none | <clip-source> | [ <basic-shape> || <geometry-box> ]
// https://drafts.fxtf.org/css-masking/#propdef-clip-path
RefPtr<CSSValue> consumeClipPath(CSSParserTokenRange&, const CSSParserContext&);

} // namespace CSSPropertyParserHelpers
} // namespace WebCore
