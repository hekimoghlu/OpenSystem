/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 10, 2024.
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
#include "SourceTaintedOrigin.h"

#include "CodeBlock.h"
#include "StackVisitor.h"
#include "VM.h"

namespace JSC {

String sourceTaintedOriginToString(SourceTaintedOrigin taintedness)
{
    switch (taintedness) {
    case SourceTaintedOrigin::Untainted: return "Untainted"_s;
    case SourceTaintedOrigin::KnownTainted: return "KnownTainted"_s;
    case SourceTaintedOrigin::IndirectlyTainted: return "IndirectlyTainted"_s;
    case SourceTaintedOrigin::IndirectlyTaintedByHistory: return "IndirectlyTaintedByHistory"_s;
    default: break;
    }
    RELEASE_ASSERT_NOT_REACHED();
    return { };
}

std::pair<SourceTaintedOrigin, URL> sourceTaintedOriginFromStack(VM& vm, CallFrame* callFrame)
{
    if (!vm.mightBeExecutingTaintedCode())
        return { SourceTaintedOrigin::Untainted, { } };
    SourceTaintedOrigin result = SourceTaintedOrigin::IndirectlyTaintedByHistory;

    URL sourceURL;
    StackVisitor::visit(callFrame, vm, [&] (StackVisitor& visitor) -> IterationStatus {
        if (!visitor->codeBlock() || !visitor->codeBlock()->couldBeTainted())
            return IterationStatus::Continue;

        auto* sourceProvider = visitor->codeBlock()->source().provider();
        result = std::max(result, sourceProvider->sourceTaintedOrigin());
        if (result != SourceTaintedOrigin::KnownTainted)
            return IterationStatus::Continue;

        sourceURL = sourceProvider->sourceOrigin().url();
        return IterationStatus::Done;
    });

    return { result, WTFMove(sourceURL) };
}

SourceTaintedOrigin computeNewSourceTaintedOriginFromStack(VM& vm, CallFrame* callFrame)
{
    if (!vm.mightBeExecutingTaintedCode())
        return SourceTaintedOrigin::Untainted;

    SourceTaintedOrigin result = SourceTaintedOrigin::IndirectlyTaintedByHistory;
    StackVisitor::visit(callFrame, vm, [&] (StackVisitor& visitor) -> IterationStatus {
        if (visitor->codeBlock() && visitor->codeBlock()->couldBeTainted()) {
            SourceTaintedOrigin currentTaintedOrigin = visitor->codeBlock()->source().provider()->sourceTaintedOrigin();
            if (currentTaintedOrigin >= SourceTaintedOrigin::IndirectlyTainted) {
                result = SourceTaintedOrigin::IndirectlyTainted;
                return IterationStatus::Done;
            }
        }

        return IterationStatus::Continue;
    });

    return result;
}

} // namespace JSC
