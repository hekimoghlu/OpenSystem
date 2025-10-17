/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 14, 2025.
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
#include "ProfilerOSRExit.h"

#include "JSGlobalObject.h"
#include "ObjectConstructor.h"
#include "ProfilerDumper.h"

namespace JSC { namespace Profiler {

OSRExit::OSRExit(unsigned id, const OriginStack& origin, ExitKind kind, bool isWatchpoint)
    : m_origin(origin)
    , m_id(id)
    , m_exitKind(kind)
    , m_isWatchpoint(isWatchpoint)
    , m_counter(0)
{
}

OSRExit::~OSRExit() = default;

Ref<JSON::Value> OSRExit::toJSON(Dumper& dumper) const
{
    auto result = JSON::Object::create();
    result->setDouble(dumper.keys().m_id, m_id);
    result->setValue(dumper.keys().m_origin, m_origin.toJSON(dumper));
    result->setString(dumper.keys().m_exitKind, enumName(m_exitKind));
    result->setBoolean(dumper.keys().m_isWatchpoint, !!m_isWatchpoint);
    result->setDouble(dumper.keys().m_count, m_counter);
    return result;
}

} } // namespace JSC::Profiler

