/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 24, 2023.
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

#include "CSSXywhFunction.h"
#include "StyleInsetFunction.h"

namespace WebCore {
namespace Style {

// NOTE: There is no Style::Xywh type. Style conversion of CSS::Xywh yields a Style::Inset.
template<> struct ToStyle<CSS::Xywh> { auto operator()(const CSS::Xywh&, const BuilderState&) -> Inset; };
template<> struct ToStyle<CSS::XywhFunction> { auto operator()(const CSS::XywhFunction&, const BuilderState&) -> InsetFunction; };

} // namespace Style
} // namespace WebCore
