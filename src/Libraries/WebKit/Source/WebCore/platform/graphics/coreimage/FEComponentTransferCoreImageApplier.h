/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 22, 2024.
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

#if USE(CORE_IMAGE)

#import "FilterEffectApplier.h"
#import <wtf/TZoneMalloc.h>

namespace WebCore {

class FEComponentTransfer;

class FEComponentTransferCoreImageApplier final : public FilterEffectConcreteApplier<FEComponentTransfer> {
    WTF_MAKE_TZONE_ALLOCATED(FEComponentTransferCoreImageApplier);
    using Base = FilterEffectConcreteApplier<FEComponentTransfer>;

public:
    FEComponentTransferCoreImageApplier(const FEComponentTransfer&);

    static bool supportsCoreImageRendering(const FEComponentTransfer&);

private:
    bool apply(const Filter&, const FilterImageVector& inputs, FilterImage& result) const final;
};

} // namespace WebCore

#endif // USE(CORE_IMAGE)
