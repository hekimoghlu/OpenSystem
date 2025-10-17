/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 15, 2023.
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
#include <wtf/CodePtr.h>

#include <wtf/PrintStream.h>

namespace WTF {

void CodePtrBase::dumpWithName(void* executableAddress, void* dataLocation, ASCIILiteral name, PrintStream& out)
{
    if (!executableAddress) {
        out.print(name, "(null)"_s);
        return;
    }
    if (executableAddress == dataLocation) {
        out.print(name, "("_s, RawPointer(executableAddress), ")"_s);
        return;
    }
    out.print(name, "(executable = "_s, RawPointer(executableAddress), ", dataLocation = "_s, RawPointer(dataLocation), ")"_s);
}

} // namespace WTF
