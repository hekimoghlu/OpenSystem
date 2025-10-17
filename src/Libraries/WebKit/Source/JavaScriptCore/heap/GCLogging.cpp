/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 19, 2023.
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
#include "GCLogging.h"

#include <wtf/PrintStream.h>


namespace WTF {

void printInternal(PrintStream& out, JSC::GCLogging::Level level)
{
    switch (level) {
    case JSC::GCLogging::Level::None:
        out.print("None");
        return;
    case JSC::GCLogging::Level::Basic:
        out.print("Basic");
        return;
    case JSC::GCLogging::Level::Verbose:
        out.print("Verbose");
        return;
    default:
        out.print("Level=", level - JSC::GCLogging::Level::None);
        return;
    }
}

} // namespace WTF

