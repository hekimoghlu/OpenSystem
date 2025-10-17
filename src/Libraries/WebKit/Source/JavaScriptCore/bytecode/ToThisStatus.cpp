/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 20, 2022.
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
#include "ToThisStatus.h"

namespace JSC {

ToThisStatus merge(ToThisStatus a, ToThisStatus b)
{
    switch (a) {
    case ToThisOK:
        return b;
    case ToThisConflicted:
        return ToThisConflicted;
    case ToThisClearedByGC:
        if (b == ToThisConflicted)
            return ToThisConflicted;
        return ToThisClearedByGC;
    }
    
    RELEASE_ASSERT_NOT_REACHED();
    return ToThisConflicted;
}

} // namespace JSC

namespace WTF {

using namespace JSC;

void printInternal(PrintStream& out, ToThisStatus status)
{
    switch (status) {
    case ToThisOK:
        out.print("OK");
        return;
    case ToThisConflicted:
        out.print("Conflicted");
        return;
    case ToThisClearedByGC:
        out.print("ClearedByGC");
        return;
    }
    
    RELEASE_ASSERT_NOT_REACHED();
}

} // namespace WTF

