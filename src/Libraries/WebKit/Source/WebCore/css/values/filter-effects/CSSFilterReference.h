/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 15, 2025.
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

#include "CSSValueTypes.h"
#include <wtf/text/WTFString.h>

namespace WebCore {
namespace CSS {

// https://drafts.fxtf.org/filter-effects/#typedef-filter-url
struct FilterReference {
    String url;

    bool operator==(const FilterReference&) const = default;
};

template<> struct Serialize<FilterReference> { void operator()(StringBuilder&, const FilterReference&); };
template<> struct ComputedStyleDependenciesCollector<FilterReference> { constexpr void operator()(ComputedStyleDependencies&, const FilterReference&) { } };
template<> struct CSSValueChildrenVisitor<FilterReference> { constexpr IterationStatus operator()(const Function<IterationStatus(CSSValue&)>&, const FilterReference&) { return IterationStatus::Continue; } };

} // namespace CSS
} // namespace WebCore
