/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 14, 2023.
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

#include "CodeBlockHash.h"
#include "JSCJSValue.h"
#include "ProfilerBytecodeSequence.h"
#include <wtf/PrintStream.h>
#include <wtf/text/WTFString.h>

namespace JSC { namespace Profiler {

class Bytecodes : public BytecodeSequence {
public:
    Bytecodes(size_t id, CodeBlock*);
    ~Bytecodes();
    
    size_t id() const { return m_id; }
    const CString& inferredName() const { return m_inferredName; }
    const CString& sourceCode() const { return m_sourceCode; }
    unsigned instructionCount() const { return m_instructionCount; }
    CodeBlockHash hash() const { return m_hash; }

    void dump(PrintStream&) const;
    
    Ref<JSON::Value> toJSON(Dumper&) const;
    
private:
    size_t m_id;
    CString m_inferredName;
    CString m_sourceCode;
    CodeBlockHash m_hash;
    unsigned m_instructionCount;
};

} } // namespace JSC::Profiler
