/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 20, 2024.
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
#include "JITCompilationMode.h"

#if ENABLE(JIT)

namespace WTF {

using namespace JSC;

void printInternal(PrintStream& out, JITCompilationMode mode)
{
    switch (mode) {
    case JITCompilationMode::InvalidCompilation:
        out.print("InvalidCompilation");
        return;
    case JITCompilationMode::Baseline:
        out.print("Baseline");
        return;
    case JITCompilationMode::DFG:
        out.print("DFG");
        return;
    case JITCompilationMode::UnlinkedDFG:
        out.print("UnlinkedDFG");
        return;
    case JITCompilationMode::FTL:
        out.print("FTL");
        return;
    case JITCompilationMode::FTLForOSREntry:
        out.print("FTLForOSREntry");
        return;
    }
    RELEASE_ASSERT_NOT_REACHED();
}

} // namespace WTF

#endif // ENABLE(JIT)
