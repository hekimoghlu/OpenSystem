/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 6, 2022.
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

#include "AirBasicBlock.h"
#include "MacroAssembler.h"
#include <wtf/Box.h>
#include <wtf/IndexMap.h>
#include <wtf/SharedTask.h>
#include <wtf/Vector.h>

namespace JSC { namespace B3 { namespace Air {

class Code;

struct GenerationContext {
    WTF_MAKE_NONCOPYABLE(GenerationContext);
public:

    GenerationContext() = default;

    typedef void LatePathFunction(CCallHelpers&, GenerationContext&);
    typedef SharedTask<LatePathFunction> LatePath;

    Vector<RefPtr<LatePath>> latePaths;
    IndexMap<BasicBlock*, Box<MacroAssembler::Label>> blockLabels;
    BasicBlock* currentBlock { nullptr };
    unsigned indexInBlock { UINT_MAX };
    Code* code { nullptr };
};

} } } // namespace JSC::B3::Air

#endif // ENABLE(B3_JIT)
