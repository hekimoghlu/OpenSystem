/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 30, 2023.
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

#include <wtf/PrintStream.h>
#include <wtf/StdLibExtras.h>

namespace JSC { namespace DFG {

// Entries in the NodeType enum (below) are composed of an id, a result type (possibly none)
// and some additional informative flags (must generate, is constant, etc).
#define NodeResultMask                   0x0007
#define NodeResultJS                     0x0001
#define NodeResultNumber                 0x0002
#define NodeResultDouble                 0x0003
#define NodeResultInt32                  0x0004
#define NodeResultInt52                  0x0005
#define NodeResultBoolean                0x0006
#define NodeResultStorage                0x0007

#define NodeMustGenerate                 0x0008 // set on nodes that have side effects, and may not trivially be removed by DCE.
#define NodeHasVarArgs                   0x0010

#define NodeMayHaveDoubleResult          0x00020
#define NodeMayOverflowInt52             0x00040
#define NodeMayOverflowInt32InBaseline   0x00080
#define NodeMayOverflowInt32InDFG        0x00100
#define NodeMayNegZeroInBaseline         0x00200
#define NodeMayNegZeroInDFG              0x00400
#define NodeMayHaveBigInt32Result        0x00800
#define NodeMayHaveHeapBigIntResult      0x01000
#define NodeMayHaveNonNumericResult      0x02000
#define NodeBehaviorMask                 (NodeMayHaveDoubleResult | NodeMayOverflowInt52 | NodeMayOverflowInt32InBaseline | NodeMayOverflowInt32InDFG | NodeMayNegZeroInBaseline | NodeMayNegZeroInDFG | NodeMayHaveBigInt32Result | NodeMayHaveHeapBigIntResult | NodeMayHaveNonNumericResult)
#define NodeMayHaveNonIntResult          (NodeMayHaveDoubleResult | NodeMayHaveNonNumericResult | NodeMayHaveBigInt32Result | NodeMayHaveHeapBigIntResult)

#define NodeBytecodeUseBottom            0x00000
#define NodeBytecodeUsesAsNumber         0x04000 // The result of this computation may be used in a context that observes fractional, or bigger-than-int32, results.
#define NodeBytecodeNeedsNegZero         0x08000 // The result of this computation may be used in a context that observes -0.
#define NodeBytecodeNeedsNaNOrInfinity   0x10000 // The result of this computation may be used in a context that observes NaN or Infinity.
#define NodeBytecodeUsesAsOther          0x20000 // The result of this computation may be used in a context that distinguishes between NaN and other things (like undefined).
#define NodeBytecodeUsesAsInt            0x40000 // The result of this computation is known to be used in a context that prefers, but does not require, integer values.
#define NodeBytecodePrefersArrayIndex    0x80000 // The result of this computation is known to be used in a context that strongly prefers integer values, to the point that we should avoid using doubles if at all possible.
#define NodeBytecodeUsesAsArrayIndex     (NodeBytecodeUsesAsNumber | NodeBytecodeNeedsNaNOrInfinity | NodeBytecodeUsesAsOther | NodeBytecodeUsesAsInt | NodeBytecodePrefersArrayIndex)
#define NodeBytecodeUsesAsValue          (NodeBytecodeUsesAsNumber | NodeBytecodeNeedsNegZero | NodeBytecodeNeedsNaNOrInfinity | NodeBytecodeUsesAsOther)
#define NodeBytecodeBackPropMask         (NodeBytecodeUsesAsNumber | NodeBytecodeNeedsNegZero | NodeBytecodeNeedsNaNOrInfinity | NodeBytecodeUsesAsOther | NodeBytecodeUsesAsInt | NodeBytecodePrefersArrayIndex)

#define NodeArithFlagsMask               (NodeBehaviorMask | NodeBytecodeBackPropMask)

#define NodeIsFlushed                   0x100000 // Computed by CPSRethreadingPhase, will tell you which local nodes are backwards-reachable from a Flush.

#define NodeMiscFlag1                   0x200000
#define NodeMiscFlag2                   0x400000

typedef uint32_t NodeFlags;

static inline bool bytecodeUsesAsNumber(NodeFlags flags)
{
    return !!(flags & NodeBytecodeUsesAsNumber);
}

static inline bool bytecodeCanTruncateInteger(NodeFlags flags)
{
    return !bytecodeUsesAsNumber(flags);
}

static inline bool bytecodeCanIgnoreNegativeZero(NodeFlags flags)
{
    return !(flags & NodeBytecodeNeedsNegZero);
}

static inline bool bytecodeCanIgnoreNaNAndInfinity(NodeFlags flags)
{
    return !(flags & NodeBytecodeNeedsNaNOrInfinity);
}

enum RareCaseProfilingSource {
    BaselineRareCase, // Comes from slow case counting in the baseline JIT.
    DFGRareCase, // Comes from OSR exit profiles.
    AllRareCases
};

static inline bool nodeMayOverflowInt52(NodeFlags flags, RareCaseProfilingSource)
{
    return !!(flags & NodeMayOverflowInt52);
}

static inline bool nodeMayOverflowInt32(NodeFlags flags, RareCaseProfilingSource source)
{
    NodeFlags mask = 0;
    switch (source) {
    case BaselineRareCase:
        mask = NodeMayOverflowInt32InBaseline;
        break;
    case DFGRareCase:
        mask = NodeMayOverflowInt32InDFG;
        break;
    case AllRareCases:
        mask = NodeMayOverflowInt32InBaseline | NodeMayOverflowInt32InDFG;
        break;
    }
    return !!(flags & mask);
}

static inline bool nodeMayNegZero(NodeFlags flags, RareCaseProfilingSource source)
{
    NodeFlags mask = 0;
    switch (source) {
    case BaselineRareCase:
        mask = NodeMayNegZeroInBaseline;
        break;
    case DFGRareCase:
        mask = NodeMayNegZeroInDFG;
        break;
    case AllRareCases:
        mask = NodeMayNegZeroInBaseline | NodeMayNegZeroInDFG;
        break;
    }
    return !!(flags & mask);
}

static inline bool nodeCanSpeculateInt32(NodeFlags flags, RareCaseProfilingSource source)
{
    if (nodeMayOverflowInt32(flags, source))
        return !bytecodeUsesAsNumber(flags);
    
    if (nodeMayNegZero(flags, source))
        return bytecodeCanIgnoreNegativeZero(flags);
    
    return true;
}

static inline bool nodeCanSpeculateInt52(NodeFlags flags, RareCaseProfilingSource source)
{
    if (nodeMayOverflowInt52(flags, source))
        return false;

    if (nodeMayNegZero(flags, source))
        return bytecodeCanIgnoreNegativeZero(flags);
    
    return true;
}

static inline bool nodeMayHaveHeapBigInt(NodeFlags flags, RareCaseProfilingSource)
{
    return !!(flags & NodeMayHaveHeapBigIntResult);
}

static inline bool nodeCanSpeculateBigInt32(NodeFlags flags, RareCaseProfilingSource source)
{
    // We always attempt to produce BigInt32 as much as possible. If we see HeapBigInt, we observed overflow.
    // FIXME: Extend this information by tiered profiling (from Baseline, DFG etc.).
    // https://bugs.webkit.org/show_bug.cgi?id=211039
    return !nodeMayHaveHeapBigInt(flags, source);
}

// FIXME: Get rid of this.
// https://bugs.webkit.org/show_bug.cgi?id=131689
static inline NodeFlags canonicalResultRepresentation(NodeFlags flags)
{
    switch (flags) {
    case NodeResultDouble:
    case NodeResultInt52:
    case NodeResultStorage:
        return flags;
    default:
        return NodeResultJS;
    }
}

void dumpNodeFlags(PrintStream&, NodeFlags);
MAKE_PRINT_ADAPTOR(NodeFlagsDump, NodeFlags, dumpNodeFlags);

} } // namespace JSC::DFG

#endif // ENABLE(DFG_JIT)
