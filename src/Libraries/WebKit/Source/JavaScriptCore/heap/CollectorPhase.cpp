/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 6, 2022.
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
#include "CollectorPhase.h"

#include <wtf/PrintStream.h>

namespace JSC {

bool worldShouldBeSuspended(CollectorPhase phase)
{
    switch (phase) {
    case CollectorPhase::NotRunning:
    case CollectorPhase::Concurrent:
        return false;
        
    case CollectorPhase::Begin:
    case CollectorPhase::Fixpoint:
    case CollectorPhase::Reloop:
    case CollectorPhase::End:
        return true;
    }
    
    RELEASE_ASSERT_NOT_REACHED();
    return false;
}

} // namespace JSC

namespace WTF {

using namespace JSC;

void printInternal(PrintStream& out, JSC::CollectorPhase phase)
{
    switch (phase) {
    case CollectorPhase::NotRunning:
        out.print("NotRunning");
        return;
    case CollectorPhase::Begin:
        out.print("Begin");
        return;
    case CollectorPhase::Fixpoint:
        out.print("Fixpoint");
        return;
    case CollectorPhase::Concurrent:
        out.print("Concurrent");
        return;
    case CollectorPhase::Reloop:
        out.print("Reloop");
        return;
    case CollectorPhase::End:
        out.print("End");
        return;
    }
    
    RELEASE_ASSERT_NOT_REACHED();
}

} // namespace WTF

