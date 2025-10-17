/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 4, 2024.
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

#if ENABLE(DFG_JIT)

#include "DFGFlushFormat.h"
#include "VirtualRegister.h"

namespace JSC { namespace DFG {

class FlushedAt {
public:
    FlushedAt()
        : m_format(DeadFlush)
    {
    }
    
    explicit FlushedAt(FlushFormat format)
        : m_format(format)
    {
        ASSERT(format == DeadFlush || format == ConflictingFlush);
    }
    
    FlushedAt(FlushFormat format, VirtualRegister virtualRegister)
        : m_format(format)
        , m_virtualRegister(virtualRegister)
    {
        if (format == DeadFlush)
            ASSERT(!virtualRegister.isValid());
    }
    
    bool operator!() const { return m_format == DeadFlush; }
    
    FlushFormat format() const { return m_format; }
    VirtualRegister virtualRegister() const { return m_virtualRegister; }
    
    friend bool operator==(const FlushedAt&, const FlushedAt&) = default;
    
    FlushedAt merge(const FlushedAt& other) const
    {
        if (!*this)
            return other;
        if (!other)
            return *this;
        if (*this == other)
            return *this;
        return FlushedAt(ConflictingFlush);
    }
    
    void dump(PrintStream&) const;
    void dumpInContext(PrintStream&, DumpContext*) const;
    
private:
    FlushFormat m_format;
    VirtualRegister m_virtualRegister;
};

} } // namespace JSC::DFG

#endif // ENABLE(DFG_JIT)
