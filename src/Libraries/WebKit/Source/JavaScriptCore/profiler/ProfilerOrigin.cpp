/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 13, 2023.
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
#include "ProfilerOrigin.h"

#include "JSCInlines.h"
#include "ObjectConstructor.h"
#include "ProfilerBytecodes.h"
#include "ProfilerDatabase.h"
#include "ProfilerDumper.h"

namespace JSC { namespace Profiler {

Origin::Origin(Database& database, CodeBlock* codeBlock, BytecodeIndex bytecodeIndex)
    : m_bytecodes(database.ensureBytecodesFor(codeBlock))
    , m_bytecodeIndex(bytecodeIndex)
{
}

void Origin::dump(PrintStream& out) const
{
    out.print(*m_bytecodes, " ", m_bytecodeIndex);
}

Ref<JSON::Value> Origin::toJSON(Dumper& dumper) const
{
    auto result = JSON::Object::create();
    result->setDouble(dumper.keys().m_bytecodesID, m_bytecodes->id());
    result->setDouble(dumper.keys().m_bytecodeIndex, m_bytecodeIndex.offset());
    return result;
}

} } // namespace JSC::Profiler

