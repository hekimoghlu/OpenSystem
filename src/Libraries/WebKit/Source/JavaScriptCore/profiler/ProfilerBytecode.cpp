/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 30, 2025.
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
#include "ProfilerBytecode.h"

#include "JSCInlines.h"
#include "ObjectConstructor.h"
#include "Opcode.h"
#include "ProfilerDumper.h"

WTF_ALLOW_UNSAFE_BUFFER_USAGE_BEGIN

namespace JSC { namespace Profiler {

Ref<JSON::Value> Bytecode::toJSON(Dumper& dumper) const
{
    auto result = JSON::Object::create();
    result->setDouble(dumper.keys().m_bytecodeIndex, m_bytecodeIndex);
    result->setString(dumper.keys().m_opcode, String::fromUTF8(opcodeNames[m_opcodeID]));
    result->setString(dumper.keys().m_description, String::fromUTF8(m_description.span()));
    return result;
}

} } // namespace JSC::Profiler

WTF_ALLOW_UNSAFE_BUFFER_USAGE_END
