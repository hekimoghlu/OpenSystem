/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 19, 2021.
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

#include <wtf/PrintStream.h>

namespace JSC {

enum class ConstraintVolatility : uint8_t {
    // The constraint needs to be validated, but it is unlikely to ever produce information.
    // It's best to run it at the bitter end.
    SeldomGreyed,
    
    // FIXME: We could introduce a new kind of volatility called GreyedByResumption, which
    // would mean running all of the times that GreyedByExecution runs except as a root in a
    // full GC.
    // https://bugs.webkit.org/show_bug.cgi?id=166830
    
    // The constraint needs to be reevaluated anytime the mutator runs: so at GC start and
    // whenever the GC resuspends after a resumption. This is almost always something that
    // you'd call a "root" in a traditional GC.
    GreyedByExecution,
    
    // The constraint needs to be reevaluated any time any object is marked and anytime the
    // mutator resumes.
    GreyedByMarking
};
    
} // namespace JSC

namespace WTF {

inline void printInternal(PrintStream& out, JSC::ConstraintVolatility volatility)
{
    switch (volatility) {
    case JSC::ConstraintVolatility::SeldomGreyed:
        out.print("SeldomGreyed");
        return;
    case JSC::ConstraintVolatility::GreyedByExecution:
        out.print("GreyedByExecuction");
        return;
    case JSC::ConstraintVolatility::GreyedByMarking:
        out.print("GreyedByMarking");
        return;
    }
    RELEASE_ASSERT_NOT_REACHED();
}

} // namespace WTF

