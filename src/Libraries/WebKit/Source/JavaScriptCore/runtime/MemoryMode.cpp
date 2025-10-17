/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 16, 2022.
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
#include "MemoryMode.h"

#include <wtf/Assertions.h>
#include <wtf/DataLog.h>
#include <wtf/PrintStream.h>

namespace WTF {

void printInternal(PrintStream& out, JSC::MemoryMode mode)
{
    switch (mode) {
    case JSC::MemoryMode::BoundsChecking:
        out.print("BoundsChecking");
        return;
    case JSC::MemoryMode::Signaling:
        out.print("Signaling");
        return;
    }
    RELEASE_ASSERT_NOT_REACHED();
}

void printInternal(PrintStream& out, JSC::MemorySharingMode mode)
{
    switch (mode) {
    case JSC::MemorySharingMode::Default:
        out.print("Default");
        return;
    case JSC::MemorySharingMode::Shared:
        out.print("Shared");
        return;
    }
    RELEASE_ASSERT_NOT_REACHED();
}

} // namespace WTF
