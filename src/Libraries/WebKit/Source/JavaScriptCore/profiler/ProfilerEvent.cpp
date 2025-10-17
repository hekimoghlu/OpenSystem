/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 25, 2025.
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
#include "ProfilerEvent.h"

#include "JSCInlines.h"
#include "ObjectConstructor.h"
#include "ProfilerBytecodes.h"
#include "ProfilerCompilation.h"
#include "ProfilerDumper.h"
#include "ProfilerUID.h"

namespace JSC { namespace Profiler {

void Event::dump(PrintStream& out) const
{
    out.print(m_time, ": ", pointerDump(m_bytecodes));
    if (m_compilation)
        out.print(" ", *m_compilation);
    out.print(": ", m_summary);
    if (m_detail.length())
        out.print(" (", m_detail, ")");
}

Ref<JSON::Value> Event::toJSON(Dumper& dumper) const
{
    auto result = JSON::Object::create();

    result->setDouble(dumper.keys().m_time, m_time.secondsSinceEpoch().value());
    result->setDouble(dumper.keys().m_bytecodesID, m_bytecodes->id());
    if (m_compilation)
        result->setString(dumper.keys().m_compilationUID, makeString(m_compilation->uid()));
    result->setString(dumper.keys().m_summary, String::fromUTF8(m_summary));
    if (m_detail.length())
        result->setString(dumper.keys().m_detail, String::fromUTF8(m_detail.span()));

    return result;
}

} } // namespace JSC::Profiler

