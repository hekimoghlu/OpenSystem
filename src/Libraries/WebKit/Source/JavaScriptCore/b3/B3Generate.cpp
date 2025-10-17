/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 13, 2024.
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
#include "B3Generate.h"

#if ENABLE(B3_JIT)

#include "AirGenerate.h"
#include "B3CanonicalizePrePostIncrements.h"
#include "B3Common.h"
#include "B3DuplicateTails.h"
#include "B3EliminateCommonSubexpressions.h"
#include "B3EliminateDeadCode.h"
#include "B3FixSSA.h"
#include "B3FoldPathConstants.h"
#include "B3HoistLoopInvariantValues.h"
#include "B3InferSwitches.h"
#include "B3LegalizeMemoryOffsets.h"
#include "B3LowerInt64.h"
#include "B3LowerMacros.h"
#include "B3LowerMacrosAfterOptimizations.h"
#include "B3LowerToAir.h"
#include "B3MoveConstants.h"
#include "B3OptimizeAssociativeExpressionTrees.h"
#include "B3Procedure.h"
#include "B3ReduceDoubleToFloat.h"
#include "B3ReduceStrength.h"
#include "B3Validate.h"
#include "CompilerTimingScope.h"

namespace JSC { namespace B3 {

void prepareForGeneration(Procedure& procedure)
{
    CompilerTimingScope timingScope("Total B3+Air"_s, "prepareForGeneration"_s);

    generateToAir(procedure);
    Air::prepareForGeneration(procedure.code());
}

void generate(Procedure& procedure, CCallHelpers& jit)
{
    Air::generate(procedure.code(), jit);
}

void generateToAir(Procedure& procedure)
{
    CompilerTimingScope timingScope("Total B3"_s, "generateToAir"_s);
    
    if ((shouldDumpIR(procedure, B3Mode) || Options::dumpGraphAfterParsing()) && !shouldDumpIRAtEachPhase(B3Mode)) {
        dataLog(tierName, "Initial B3:\n");
        dataLog(procedure);
    }

    // We don't require the incoming IR to have predecessors computed.
    procedure.resetReachability();
    
    if (shouldValidateIR())
        validate(procedure);
    
    if (procedure.optLevel() >= 2) {
        reduceDoubleToFloat(procedure);
        reduceStrength(procedure);
        if (Options::useB3HoistLoopInvariantValues())
            hoistLoopInvariantValues(procedure);
        if (eliminateCommonSubexpressions(procedure))
            eliminateCommonSubexpressions(procedure);
        eliminateDeadCode(procedure);
        inferSwitches(procedure);
        if (Options::useB3TailDup())
            duplicateTails(procedure);
        fixSSA(procedure);
        foldPathConstants(procedure);
        // FIXME: Add more optimizations here.
        // https://bugs.webkit.org/show_bug.cgi?id=150507
    } else if (procedure.optLevel() >= 1) {
        // FIXME: Explore better "quick mode" optimizations.
        reduceStrength(procedure);
    }

    // This puts the IR in quirks mode.
    lowerMacros(procedure);

    if (procedure.optLevel() >= 2) {
        optimizeAssociativeExpressionTrees(procedure);
        reduceStrength(procedure);

        // FIXME: Add more optimizations here.
        // https://bugs.webkit.org/show_bug.cgi?id=150507
    }
#if USE(JSVALUE32_64)
    lowerInt64(procedure);
#endif

    lowerMacrosAfterOptimizations(procedure);
    legalizeMemoryOffsets(procedure);
    moveConstants(procedure);
    legalizeMemoryOffsets(procedure);
    eliminateDeadCode(procedure);
    if (Options::useB3CanonicalizePrePostIncrements() && procedure.optLevel() >= 2)
        canonicalizePrePostIncrements(procedure);

    // FIXME: We should run pureCSE here to clean up some platform specific changes from the previous phases.
    // https://bugs.webkit.org/show_bug.cgi?id=164873

    if (shouldValidateIR())
        validate(procedure);
    
    // If we're doing super verbose dumping, the phase scope of any phase will already do a dump.
    // Note that lowerToAir() acts like a phase in this regard.
    if (shouldDumpIR(procedure, B3Mode) && !shouldDumpIRAtEachPhase(B3Mode)) {
        dataLog("B3 after ", procedure.lastPhaseName(), ", before generation:\n");
        dataLog(procedure);
    }

    lowerToAir(procedure);
    if (shouldDumpIR(procedure, B3Mode))
        procedure.setShouldDumpIR();
    procedure.freeUnneededB3ValuesAfterLowering();
}

} } // namespace JSC::B3

#endif // ENABLE(B3_JIT)

