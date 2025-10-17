/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 23, 2021.
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

#include "Options.h"
#include <wtf/text/ASCIILiteral.h>

namespace JSC {

class CallFrame;
class JSGlobalObject;
class ThrowScope;

// Call this only if you know that exception fuzzing is enabled.
void doExceptionFuzzing(JSGlobalObject*, ThrowScope&, ASCIILiteral where, const void* returnPC);

// This is what you should call if you don't know if fuzzing is enabled.
ALWAYS_INLINE void doExceptionFuzzingIfEnabled(JSGlobalObject* globalObject, ThrowScope& scope, ASCIILiteral where, const void* returnPC)
{
    if (LIKELY(!Options::useExceptionFuzz()))
        return;
    doExceptionFuzzing(globalObject, scope, where, returnPC);
}

} // namespace JSC
