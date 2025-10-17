/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 27, 2024.
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

#include "JSDOMBinding.h"
#include "JSElement.h"

namespace JSC {
namespace JSCastingHelpers {

template<>
struct InheritsTraits<WebCore::JSElement> {
    static constexpr std::optional<JSTypeRange> typeRange { JSTypeRange { static_cast<JSType>(WebCore::JSElementType), static_cast<JSType>(WebCore::JSElementType) } };
    template<typename From>
    static inline bool inherits(From* from)
    {
        return inheritsJSTypeImpl<WebCore::JSElement>(from, *typeRange);
    }
};

} // namespace JSCastingHelpers
} // namespace JSC
