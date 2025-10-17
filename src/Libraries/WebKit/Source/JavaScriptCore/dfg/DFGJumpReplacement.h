/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 12, 2025.
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

#if ENABLE(DFG_JIT)

#include "CodeLocation.h"

namespace JSC { namespace DFG {

class JumpReplacement {
public:
    JumpReplacement(CodeLocationLabel<JSInternalPtrTag> source, CodeLocationLabel<OSRExitPtrTag> destination)
        : m_source(source)
        , m_destination(destination)
    {
    }
    
    void fire();
    void installVMTrapBreakpoint();
    void* dataLocation() const { return m_source.dataLocation(); }

private:
    CodeLocationLabel<JSInternalPtrTag> m_source;
    CodeLocationLabel<OSRExitPtrTag> m_destination;
};

} } // namespace JSC::DFG

#endif // ENABLE(DFG_JIT)
