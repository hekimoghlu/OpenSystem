/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 26, 2021.
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
#include "GCConductor.h"

#include <wtf/PrintStream.h>

namespace JSC {

const char* gcConductorShortName(GCConductor conn)
{
    switch (conn) {
    case GCConductor::Mutator:
        return "M";
    case GCConductor::Collector:
        return "C";
    }
    
    RELEASE_ASSERT_NOT_REACHED();
}

} // namespace JSC

namespace WTF {

using namespace JSC;

void printInternal(PrintStream& out, GCConductor conn)
{
    switch (conn) {
    case GCConductor::Mutator:
        out.print("Mutator");
        return;
    case GCConductor::Collector:
        out.print("Collector");
        return;
    }
    
    RELEASE_ASSERT_NOT_REACHED();
}

} // namespace WTF

