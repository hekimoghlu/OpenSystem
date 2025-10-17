/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 20, 2024.
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
#include "PredictionFileCreatingFuzzerAgent.h"
#include <wtf/DataLog.h>
#include <wtf/TZoneMallocInlines.h>

WTF_ALLOW_UNSAFE_BUFFER_USAGE_BEGIN

namespace JSC {

WTF_MAKE_TZONE_ALLOCATED_IMPL(PredictionFileCreatingFuzzerAgent);

PredictionFileCreatingFuzzerAgent::PredictionFileCreatingFuzzerAgent(VM& vm)
    : FileBasedFuzzerAgentBase(vm)
{
}

SpeculatedType PredictionFileCreatingFuzzerAgent::getPredictionInternal(CodeBlock*, PredictionTarget& predictionTarget, SpeculatedType original)
{
    switch (predictionTarget.opcodeId) {
    case op_to_this:
    case op_get_by_val:
    case op_get_argument:
    case op_get_from_arguments:
    case op_get_from_scope:
    case op_get_by_id:
    case op_get_length:
    case op_get_by_id_with_this:
    case op_get_by_val_with_this:
    case op_enumerator_get_by_val:
    case op_construct:
    case op_construct_varargs:
    case op_super_construct:
    case op_super_construct_varargs:
    case op_call:
    case op_call_ignore_result:
    case op_call_direct_eval:
    case op_call_varargs:
    case op_tail_call:
    case op_tail_call_varargs:
        dataLogF("%s:%" PRIx64 "\n", predictionTarget.lookupKey.utf8().data(), original);
        break;

    default:
        RELEASE_ASSERT_WITH_MESSAGE(false, "unhandled opcode: %s", opcodeNames[predictionTarget.opcodeId].characters());
    }
    return original;
}

} // namespace JSC

WTF_ALLOW_UNSAFE_BUFFER_USAGE_END
