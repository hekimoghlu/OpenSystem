/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 11, 2024.
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
#include "ExceptionFuzz.h"

#include "Error.h"
#include "HeapInlines.h"
#include "JSCJSValueInlines.h"
#include "TestRunnerUtils.h"

namespace JSC {

static unsigned s_numberOfExceptionFuzzChecks;
unsigned numberOfExceptionFuzzChecks() { return s_numberOfExceptionFuzzChecks; }

// Call this only if you know that exception fuzzing is enabled.
void doExceptionFuzzing(JSGlobalObject* globalObject, ThrowScope& scope, ASCIILiteral where, const void* returnPC)
{
    VM& vm = scope.vm();
    ASSERT(Options::useExceptionFuzz());

    DeferGCForAWhile deferGC(vm);
    
    s_numberOfExceptionFuzzChecks++;
    
    unsigned fireTarget = Options::fireExceptionFuzzAt();
    if (fireTarget == s_numberOfExceptionFuzzChecks) {
        SAFE_PRINTF("JSC EXCEPTION FUZZ: Throwing fuzz exception with call frame %p, seen in %s and return address %p.\n", globalObject, where, returnPC);
        fflush(stdout);

        // The ThrowScope also checks for unchecked simulated exceptions before throwing a
        // new exception. This ensures that we don't quietly overwrite a pending exception
        // (which should never happen with the only exception being to rethrow the same
        // exception). However, ExceptionFuzz works by intentionally throwing its own exception
        // even when one may already exist. This is ok for ExceptionFuzz testing, but we need
        // to placate the exception check verifier here.
        EXCEPTION_ASSERT(scope.exception() || !scope.exception());

        throwException(globalObject, scope, createError(globalObject, "Exception Fuzz"_s));
    }
}

} // namespace JSC


