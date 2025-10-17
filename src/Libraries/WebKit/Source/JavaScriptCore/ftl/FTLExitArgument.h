/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 6, 2022.
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

#if ENABLE(FTL_JIT)

#include "DataFormat.h"
#include <wtf/PrintStream.h>

namespace JSC { namespace FTL {

struct ExitArgumentRepresentation {
    DataFormat format;
    unsigned argument;
};

class ExitArgument {
public:
    ExitArgument()
    {
        m_representation.format = DataFormatNone;
    }
    
    ExitArgument(DataFormat format, unsigned argument)
    {
        m_representation.format = format;
        m_representation.argument = argument;
    }
    
    explicit ExitArgument(ExitArgumentRepresentation representation)
    {
        m_representation = representation;
    }
    
    bool operator!() const { return m_representation.format == DataFormatNone; }
    
    DataFormat format() const
    {
        ASSERT(*this);
        return m_representation.format;
    }
    
    unsigned argument() const
    {
        ASSERT(*this);
        return m_representation.argument;
    }
    
    ExitArgument withFormat(DataFormat format)
    {
        return ExitArgument(format, argument());
    }

    ExitArgumentRepresentation representation() const { return m_representation; }
    
    void dump(PrintStream&) const;
    
private:
    ExitArgumentRepresentation m_representation;
};

} } // namespace JSC::FTL

#endif // ENABLE(FTL_JIT)
