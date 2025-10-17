/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 8, 2025.
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

#if ENABLE(B3_JIT)

#include "CompilerTimingScope.h"
#include <wtf/Noncopyable.h>
#include <wtf/text/CString.h>

namespace JSC { namespace B3 { namespace Air {

class Code;

class PhaseScope {
    WTF_MAKE_NONCOPYABLE(PhaseScope);
public:
    PhaseScope(Code&, ASCIILiteral name);
    ~PhaseScope(); // this does validation

private:
    Code& m_code;
    ASCIILiteral m_name;
    CompilerTimingScope m_timingScope;
    CString m_dumpBefore;
};

} } } // namespace JSC::B3::Air

#endif // ENABLE(B3_JIT)
