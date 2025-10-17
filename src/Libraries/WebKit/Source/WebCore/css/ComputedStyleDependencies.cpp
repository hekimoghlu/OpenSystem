/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 16, 2024.
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
#include "config.h"
#include "ComputedStyleDependencies.h"

#include "CSSToLengthConversionData.h"

namespace WebCore {

bool ComputedStyleDependencies::canResolveDependenciesWithConversionData(const CSSToLengthConversionData& conversionData) const
{
    if (!rootProperties.isEmpty() && !conversionData.rootStyle())
        return false;

    if (!properties.isEmpty() && !conversionData.style())
        return false;

    if (containerDimensions && !conversionData.elementForContainerUnitResolution())
        return false;

    if (viewportDimensions && !conversionData.renderView())
        return false;

    return true;
}

} // namespace WebCore
