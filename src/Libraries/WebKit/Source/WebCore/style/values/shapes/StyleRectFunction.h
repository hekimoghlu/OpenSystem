/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 14, 2025.
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

#include "CSSRectFunction.h"
#include "StyleInsetFunction.h"

namespace WebCore {
namespace Style {

// NOTE: There is no Style::Rect type. Style conversion of CSS::Rect yields a Style::Inset.
template<> struct ToStyle<CSS::Rect> { auto operator()(const CSS::Rect&, const BuilderState&) -> Inset; };
template<> struct ToStyle<CSS::RectFunction> { auto operator()(const CSS::RectFunction&, const BuilderState&) -> InsetFunction; };

} // namespace Style
} // namespace WebCore
