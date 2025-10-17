/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 21, 2024.
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

#include "ProfilerBytecode.h"
#include <wtf/Vector.h>
#include <wtf/text/CString.h>
#include <wtf/text/WTFString.h>

namespace JSC {

class CodeBlock;

namespace Profiler {

class Dumper;

class BytecodeSequence {
public:
    BytecodeSequence(CodeBlock*);
    ~BytecodeSequence();
    
    // Note that this data structure is not indexed by bytecode index.
    unsigned size() const { return m_sequence.size(); }
    const Bytecode& at(unsigned i) const { return m_sequence[i]; }

    unsigned indexForBytecodeIndex(unsigned bytecodeIndex) const;
    const Bytecode& forBytecodeIndex(unsigned bytecodeIndex) const;

protected:
    void addSequenceProperties(Dumper&, JSON::Object&) const;
    
private:
    Vector<CString> m_header;
    Vector<Bytecode> m_sequence;
};

} } // namespace JSC::Profiler
