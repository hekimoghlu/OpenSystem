/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 5, 2022.
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

#include "JSCJSValue.h"
#include "ProfilerOriginStack.h"
#include <wtf/text/CString.h>

namespace JSC { namespace Profiler {

class CompiledBytecode {
public:
    // It's valid to have an empty OriginStack, which indicates that this is some
    // sort of non-bytecode-related machine code.
    CompiledBytecode(const OriginStack&, const CString& description);
    ~CompiledBytecode();
    
    const OriginStack& originStack() const { return m_origin; }
    const CString& description() const { return m_description; }

    Ref<JSON::Value> toJSON(Dumper&) const;

private:
    OriginStack m_origin;
    CString m_description;
};

} } // namespace JSC::Profiler
