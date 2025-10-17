/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 2, 2023.
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

#include "FloatRect.h"
#include "Path.h"
#include "StyleValueTypes.h"

namespace WebCore {
namespace Style {

// All types that want to expose a generated WebCore::Path must specialize PathComputation the following member function:
//
//    template<> struct WebCore::Style::PathComputation<StyleType> {
//        WebCore::Path operator()(const StyleType&, const FloatRect&);
//    };

template<typename StyleType> struct PathComputation;

template<typename StyleType> WebCore::Path path(const StyleType& value, const FloatRect& referenceBox)
{
    return PathComputation<StyleType>{}(value, referenceBox);
}

// Specialization for `FunctionNotation`.
template<CSSValueID Name, typename StyleType> struct PathComputation<FunctionNotation<Name, StyleType>> {
    WebCore::Path operator()(const FunctionNotation<Name, StyleType>& value, const FloatRect& referenceBox)
    {
        return WebCore::Style::path(value.parameters, referenceBox);
    }
};

} // namespace Style
} // namespace WebCore
