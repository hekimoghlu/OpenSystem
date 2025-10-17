/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 2, 2025.
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
#include "FileBasedFuzzerAgent.h"

#include "CodeBlock.h"
#include "FuzzerPredictions.h"
#include "JSCellInlines.h"
#include <wtf/AnsiColors.h>
#include <wtf/TZoneMallocInlines.h>

WTF_ALLOW_UNSAFE_BUFFER_USAGE_BEGIN

namespace JSC {

WTF_MAKE_TZONE_ALLOCATED_IMPL(FileBasedFuzzerAgent);

FileBasedFuzzerAgent::FileBasedFuzzerAgent(VM& vm)
    : FileBasedFuzzerAgentBase(vm)
{
}

SpeculatedType FileBasedFuzzerAgent::getPredictionInternal(CodeBlock* codeBlock, PredictionTarget& target, SpeculatedType original)
{
    FuzzerPredictions& fuzzerPredictions = ensureGlobalFuzzerPredictions();
    std::optional<SpeculatedType> generated = fuzzerPredictions.predictionFor(target.lookupKey);

    SourceProvider* provider = codeBlock->source().provider();
    auto sourceUpToDivot = provider->source().substring(target.info.divot - target.info.startOffset, target.info.startOffset);
    auto sourceAfterDivot = provider->source().substring(target.info.divot, target.info.endOffset);

    switch (target.opcodeId) {
    // FIXME: these can not be targeted at all due to the bugs below
    case op_to_this: // broken https://bugs.webkit.org/show_bug.cgi?id=203555
    case op_get_argument: // broken https://bugs.webkit.org/show_bug.cgi?id=203554
        return original;

    // FIXME: the output of codeBlock->expressionInfoForBytecodeIndex() allows for some of
    // these opcodes to have predictions, but not all instances can be reliably targeted.
    case op_get_from_scope: // partially broken https://bugs.webkit.org/show_bug.cgi?id=203603
    case op_get_from_arguments: // partially broken https://bugs.webkit.org/show_bug.cgi?id=203608
    case op_get_by_val: // partially broken https://bugs.webkit.org/show_bug.cgi?id=203665
    case op_get_by_id: // sometimes occurs implicitly for things related to Symbol.iterator
    case op_get_length: // sometimes occurs implicitly for things related to Symbol.iterator
        if (!generated)
            return original;
        break;

    case op_call: // op_call appears implicitly in for-of loops, generators, spread/rest elements, destructuring assignment
    case op_call_ignore_result:
        if (!generated) {
            if (sourceAfterDivot.containsIgnoringASCIICase("of "_s))
                return original;
            if (sourceAfterDivot.containsIgnoringASCIICase("..."_s))
                return original;
            if (sourceAfterDivot.containsIgnoringASCIICase("yield"_s))
                return original;
            if (sourceAfterDivot.startsWith('[') && sourceAfterDivot.endsWith(']'))
                return original;
            if (sourceUpToDivot.containsIgnoringASCIICase("yield"_s))
                return original;
            if (sourceUpToDivot == "..."_s)
                return original;
            if (!target.info.startOffset && !target.info.endOffset)
                return original;
        }
        break;

    case op_get_by_val_with_this:
    case op_construct:
    case op_construct_varargs:
    case op_super_construct:
    case op_super_construct_varargs:
    case op_call_varargs:
    case op_call_direct_eval:
    case op_tail_call:
    case op_tail_call_varargs:
    case op_get_by_id_with_this:
        break;

    default:
        RELEASE_ASSERT_NOT_REACHED_WITH_MESSAGE("Unhandled opcode %s", opcodeNames[target.opcodeId].characters());
    }
    if (!generated) {
        if (Options::dumpFuzzerAgentPredictions())
            dataLogLn(MAGENTA(BOLD(target.info.instPC)), " ", BOLD(YELLOW(target.opcodeId)), " missing prediction for: ", RED(BOLD(target.lookupKey)), " ", GREEN(target.sourceFilename), ":", CYAN(target.info.lineColumn.line), ":", CYAN(target.info.lineColumn.column), " divot: ", target.info.divot, " -", target.info.startOffset, " +", target.info.endOffset, " name: '", YELLOW(codeBlock->inferredName()), "' source: '", BLUE(sourceUpToDivot), BLUE(BOLD(sourceAfterDivot)), "'");

        RELEASE_ASSERT_WITH_MESSAGE(!Options::requirePredictionForFileBasedFuzzerAgent(), "Missing expected prediction in FuzzerAgent");
        return original;
    }
    if (Options::dumpFuzzerAgentPredictions())
        dataLogLn(YELLOW(target.opcodeId), " ", CYAN(target.lookupKey), " original: ", CYAN(BOLD(SpeculationDump(original))), " generated: ", MAGENTA(BOLD(SpeculationDump(*generated))));
    return *generated;
}

} // namespace JSC

WTF_ALLOW_UNSAFE_BUFFER_USAGE_END
