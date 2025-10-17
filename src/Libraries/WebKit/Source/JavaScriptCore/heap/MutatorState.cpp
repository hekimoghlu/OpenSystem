/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 30, 2025.
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
#include "MutatorState.h"

#include <wtf/PrintStream.h>

namespace WTF {

using namespace JSC;

void printInternal(PrintStream& out, MutatorState state)
{
    switch (state) {
    case MutatorState::Running:
        out.print("Running");
        return;
    case MutatorState::Allocating:
        out.print("Allocating");
        return;
    case MutatorState::Sweeping:
        out.print("Sweeping");
        return;
    case MutatorState::Collecting:
        out.print("Collecting");
        return;
    }
    RELEASE_ASSERT_NOT_REACHED();
}

} // namespace WTF

