/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 4, 2024.
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

#include "CodeBlock.h"
#include "CodeOrigin.h"
#include <wtf/PrintStream.h>

namespace JSC {

class FullCodeOrigin {
public:
    FullCodeOrigin(CodeBlock* codeBlock, CodeOrigin codeOrigin)
        : m_codeBlock(codeBlock)
        , m_codeOrigin(codeOrigin)
    {
    }
    FullCodeOrigin() = delete;

    JS_EXPORT_PRIVATE void dump(PrintStream&) const;
    void dumpInContext(PrintStream&, DumpContext*) const;

private:
    CodeBlock* const m_codeBlock;
    const CodeOrigin m_codeOrigin;
};

} // namespace JSC
