/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 27, 2024.
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

#include "DFGPromotedHeapLocation.h"
#include "FTLExitValue.h"

namespace JSC {

class TrackedReferences;

namespace FTL {

class ExitPropertyValue {
public:
    ExitPropertyValue()
    {
    }
    
    ExitPropertyValue(DFG::PromotedLocationDescriptor location, const ExitValue& value)
        : m_location(location)
        , m_value(value)
    {
        ASSERT(!!location == !!value);
    }
    
    bool operator!() const { return !m_location; }
    
    DFG::PromotedLocationDescriptor location() const { return m_location; }
    const ExitValue& value() const { return m_value; }
    
    ExitPropertyValue withLocalsOffset(int offset) const;
    
    void dump(PrintStream& out) const;
    
    void validateReferences(const TrackedReferences&) const;

private:
    DFG::PromotedLocationDescriptor m_location;
    ExitValue m_value;
};

} } // namespace JSC::FTL

#endif // ENABLE(FTL_JIT)
