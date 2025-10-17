/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 15, 2021.
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

#include "B3HeapRange.h"
#include <wtf/PrintStream.h>

namespace JSC { namespace B3 {

struct Effects {
    // True if this cannot continue execution in the current block.
    bool terminal { false };

    // True if this value can cause execution to terminate abruptly, and that this abrupt termination is
    // observable. An example of how this gets used is to limit the hoisting of controlDependent values.
    // Note that if exitsSideways is set to true but reads is bottom, then B3 is free to assume that
    // after abrupt termination of this procedure, none of the heap will be read. That's usually false,
    // so make sure that reads corresponds to the set of things that are readable after this function
    // terminates abruptly.
    bool exitsSideways { false };

    // True if the instruction may change semantics if hoisted above some control flow. For example,
    // loads are usually control-dependent because we must assume that any control construct (either
    // a terminal like Branch or anything that exits sideways, like Check) validates whether the
    // pointer is valid. Hoisting the load above control may cause the load to trap even though it
    // would not have otherwise trapped.
    bool controlDependent { false };

    // True if this writes to the local state. Operations that write local state don't write to anything
    // in "memory" but they have a side-effect anyway. This is for modeling Upsilons, Sets, and Fences.
    // This is a way of saying: even though this operation is not a terminal, does not exit sideways,
    // and does not write to the heap, you still cannot kill this operation.
    bool writesLocalState { false };

    // True if this reads from the local state. This is only used for Phi and Get.
    bool readsLocalState { false };

    // B3 understands things about pinned registers. Therefore, it needs to know who reads them and
    // who writes them. We don't track this on a per-register basis because that would be harder and
    // we don't need it. Note that if you want to construct an immutable pinned register while also
    // having other pinned registers that are mutable, then you can use ArgumentReg. Also note that
    // nobody will stop you from making this get out-of-sync with your clobbered register sets in
    // Patchpoint. It's recommended that you err on the side of being conservative.
    // FIXME: Explore making these be RegisterSetBuilders. That's mainly hard because it would be awkward to
    // reconcile with StackmapValue's support for clobbered regs.
    // https://bugs.webkit.org/show_bug.cgi?id=163173
    bool readsPinned { false };
    bool writesPinned { false };
    
    // Memory fences cannot be reordered around each other regardless of their effects. This is flagged
    // if the operation is a memory fence.
    bool fence { false };
    
    // WARNING: The B3::hoistLoopInvariantValues() phase thinks that it understands this exhaustively. If you
    // add any new kinds of things that can be read or written, you should check that phase.

    HeapRange writes;
    HeapRange reads;
    
    static Effects none()
    {
        return Effects();
    }

    static Effects forCall()
    {
        Effects result;
        result.exitsSideways = true;
        result.controlDependent = true;
        result.writes = HeapRange::top();
        result.reads = HeapRange::top();
        result.readsPinned = true;
        result.writesPinned = true;
        result.fence = true;
        return result;
    }
    
    static Effects forCheck()
    {
        Effects result;
        result.exitsSideways = true;
        // The program could read anything after exiting, and it's on us to declare this.
        result.reads = HeapRange::top();
        return result;
    }

    bool mustExecute() const
    {
        return terminal || exitsSideways || writesLocalState || writes || writesPinned || fence;
    }

    // Returns true if reordering instructions with these respective effects would change program
    // behavior in an observable way.
    bool interferes(const Effects&) const;
    
    friend bool operator==(const Effects&, const Effects&) = default;

    JS_EXPORT_PRIVATE void dump(PrintStream& out) const;
};

} } // namespace JSC::B3

#endif // ENABLE(B3_JIT)
