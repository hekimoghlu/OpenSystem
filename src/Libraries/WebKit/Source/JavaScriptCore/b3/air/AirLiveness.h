/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 7, 2024.
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

#if ENABLE(B3_JIT)

#include "AirLivenessAdapter.h"
#include "CompilerTimingScope.h"
#include "SuperSampler.h"
#include <wtf/Liveness.h>
#include <wtf/TZoneMalloc.h>

namespace JSC { namespace B3 { namespace Air {

template<typename Adapter>
class Liveness : public WTF::Liveness<Adapter> {
    WTF_MAKE_TZONE_ALLOCATED_TEMPLATE(Liveness);
public:
    Liveness(Code& code)
        : WTF::Liveness<Adapter>(code.cfg(), code)
    {
        SuperSamplerScope samplingScope(false);
        CompilerTimingScope timingScope("Air"_s, "Liveness"_s);
        WTF::Liveness<Adapter>::compute();
    }
};

WTF_MAKE_TZONE_ALLOCATED_TEMPLATE_IMPL(template<typename Adapter>, Liveness<Adapter>);

template<Bank bank, Arg::Temperature minimumTemperature = Arg::Cold>
using TmpLiveness = Liveness<TmpLivenessAdapter<bank, minimumTemperature>>;

typedef Liveness<TmpLivenessAdapter<GP>> GPLiveness;
typedef Liveness<TmpLivenessAdapter<FP>> FPLiveness;
typedef Liveness<UnifiedTmpLivenessAdapter> UnifiedTmpLiveness;
typedef Liveness<StackSlotLivenessAdapter> StackSlotLiveness;

} } } // namespace JSC::B3::Air

#endif // ENABLE(B3_JIT)
