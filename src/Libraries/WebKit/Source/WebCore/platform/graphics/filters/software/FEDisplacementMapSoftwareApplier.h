/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 13, 2025.
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

#include "FilterEffectApplier.h"
#include <wtf/TZoneMalloc.h>

namespace WebCore {

class FEDisplacementMap;

class FEDisplacementMapSoftwareApplier final : public FilterEffectConcreteApplier<FEDisplacementMap> {
    WTF_MAKE_TZONE_ALLOCATED(FEDisplacementMapSoftwareApplier);
    using Base = FilterEffectConcreteApplier<FEDisplacementMap>;

public:
    FEDisplacementMapSoftwareApplier(const FEDisplacementMap&);

private:
    bool apply(const Filter&, const FilterImageVector& inputs, FilterImage& result) const final;

    static inline unsigned byteOffsetOfPixel(unsigned x, unsigned y, unsigned rowBytes)
    {
        const unsigned bytesPerPixel = 4;
        return x * bytesPerPixel + y * rowBytes;
    }

    int xChannelIndex() const;
    int yChannelIndex() const;
};

} // namespace WebCore
