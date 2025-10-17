/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 28, 2022.
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
#include "RootMarkReason.h"

#include <wtf/PrintStream.h>
#include <wtf/text/ASCIILiteral.h>

namespace JSC {

ASCIILiteral rootMarkReasonDescription(RootMarkReason reason)
{
#define CASE_ROOT_MARK_REASON(reason) \
    case JSC::RootMarkReason::reason: \
        return #reason""_s; \

    switch (reason) {
        FOR_EACH_ROOT_MARK_REASON(CASE_ROOT_MARK_REASON)
    }
#undef CASE_ROOT_MARK_REASON

    ASSERT_NOT_REACHED();
    return "None"_s;
}

} // namespace JSC

namespace WTF {

void printInternal(PrintStream& out, JSC::RootMarkReason reason)
{
    out.print(JSC::rootMarkReasonDescription(reason));
}

} // namespace WTF
