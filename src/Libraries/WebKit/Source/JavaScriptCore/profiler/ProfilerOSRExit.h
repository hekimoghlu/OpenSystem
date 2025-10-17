/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 26, 2023.
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
#pragma once

#include "ExitKind.h"
#include "JSCJSValue.h"
#include "ProfilerOriginStack.h"

namespace JSC { namespace Profiler {

class Database;
class Dumper;

class OSRExit {
public:
    OSRExit(unsigned id, const OriginStack&, ExitKind, bool isWatchpoint);
    ~OSRExit();
    
    unsigned id() const { return m_id; }
    const OriginStack& origin() const { return m_origin; }
    ExitKind exitKind() const { return m_exitKind; }
    bool isWatchpoint() const { return m_isWatchpoint; }
    
    uint64_t* counterAddress() { return &m_counter; }
    uint64_t count() const { return m_counter; }
    void incCount() { m_counter++; }

    Ref<JSON::Value> toJSON(Dumper&) const;

private:
    OriginStack m_origin;
    unsigned m_id;
    ExitKind m_exitKind;
    bool m_isWatchpoint;
    uint64_t m_counter;
};

} } // namespace JSC::Profiler
