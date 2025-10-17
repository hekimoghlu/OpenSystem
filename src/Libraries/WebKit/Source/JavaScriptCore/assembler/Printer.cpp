/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 20, 2023.
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
#include "Printer.h"

namespace JSC {
namespace Printer {

void printConstCharString(PrintStream& out, Context& context)
{
    const char* str = context.data.as<const char*>();
    out.print(str);
}

void printIntptr(PrintStream& out, Context& context)
{
    out.print(context.data.as<intptr_t>());
}

void printUintptr(PrintStream& out, Context& context)
{
    out.print(context.data.as<uintptr_t>());
}

void printPointer(PrintStream& out, Context& context)
{
    out.print(RawPointer(context.data.as<const void*>()));
}

void setPrinter(PrintRecord& record, CString&& string)
{
    // FIXME: It would be nice if we can release the CStringBuffer from the CString
    // and take ownership of it here instead of copying it again.
    record.data.pointer = fastStrDup(string.data());
    record.printer = printConstCharString;
}

} // namespace Printer
} // namespace JSC
