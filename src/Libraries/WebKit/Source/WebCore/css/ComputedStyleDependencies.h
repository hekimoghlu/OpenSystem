/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 21, 2024.
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

#include <wtf/Vector.h>

namespace WebCore {

class CSSToLengthConversionData;

enum CSSPropertyID : uint16_t;

struct ComputedStyleDependencies {
    Vector<CSSPropertyID> properties;
    Vector<CSSPropertyID> rootProperties;
    bool containerDimensions { false };
    bool viewportDimensions { false };
    bool anchors { false };

    bool isComputationallyIndependent() const { return properties.isEmpty() && rootProperties.isEmpty() && !containerDimensions && !anchors; }

    // Checks to see if the provided conversion data is sufficient to resolve the provided dependencies.
    bool canResolveDependenciesWithConversionData(const CSSToLengthConversionData&) const;
};

} // namespace WebCore
