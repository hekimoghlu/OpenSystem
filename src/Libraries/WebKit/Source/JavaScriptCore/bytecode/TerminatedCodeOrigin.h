/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 27, 2022.
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

#include "CodeOrigin.h"

namespace JSC {

struct InlineCallFrame;

struct TerminatedCodeOrigin {
    TerminatedCodeOrigin() { }
    
    TerminatedCodeOrigin(const CodeOrigin& codeOrigin, InlineCallFrame* terminal)
        : codeOrigin(codeOrigin)
        , terminal(terminal)
    {
    }
    
    CodeOrigin codeOrigin;
    InlineCallFrame* terminal { nullptr };
};

struct TerminatedCodeOriginHashTranslator {
    static unsigned hash(const TerminatedCodeOrigin& value)
    {
        return value.codeOrigin.approximateHash(value.terminal);
    }
    
    static bool equal(const CodeOrigin& a, const TerminatedCodeOrigin& b)
    {
        return b.codeOrigin.isApproximatelyEqualTo(a, b.terminal);
    }
};

} // namespace JSC

