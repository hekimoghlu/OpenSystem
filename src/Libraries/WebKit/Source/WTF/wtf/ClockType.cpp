/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 17, 2025.
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
#include <wtf/ClockType.h>

#include <wtf/PrintStream.h>

namespace WTF {

void printInternal(PrintStream& out, ClockType type)
{
    switch (type) {
    case ClockType::Wall:
        out.print("Wall");
        return;
    case ClockType::Monotonic:
        out.print("Monotonic");
        return;
    case ClockType::Approximate:
        out.print("Approximate");
        return;
    case ClockType::Continuous:
        out.print("Continuous");
        return;
    case ClockType::ContinuousApproximate:
        out.print("ContinuousApproximate");
        return;
    }
    RELEASE_ASSERT_NOT_REACHED();
}

} // namespace WTF

