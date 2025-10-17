/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 23, 2023.
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
#include "ProfilerProfiledBytecodes.h"

#include "JSCInlines.h"
#include "ObjectConstructor.h"
#include "ProfilerDumper.h"

namespace JSC { namespace Profiler {

ProfiledBytecodes::ProfiledBytecodes(Bytecodes* bytecodes, CodeBlock* profiledBlock)
    : BytecodeSequence(profiledBlock)
    , m_bytecodes(bytecodes)
{
}

ProfiledBytecodes::~ProfiledBytecodes() = default;

Ref<JSON::Value> ProfiledBytecodes::toJSON(Dumper& dumper) const
{
    auto result = JSON::Object::create();
    result->setDouble(dumper.keys().m_bytecodesID, m_bytecodes->id());
    addSequenceProperties(dumper, result.get());
    return result;
}

} } // namespace JSC::Profiler

