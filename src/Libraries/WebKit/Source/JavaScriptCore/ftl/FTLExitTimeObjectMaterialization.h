/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 6, 2025.
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

#include "DFGNodeType.h"
#include "FTLExitPropertyValue.h"
#include "FTLExitValue.h"
#include <wtf/Noncopyable.h>

namespace JSC {

class TrackedReferences;

namespace FTL {

class ExitTimeObjectMaterialization {
    WTF_MAKE_NONCOPYABLE(ExitTimeObjectMaterialization)
public:
    ExitTimeObjectMaterialization(DFG::NodeType, CodeOrigin);
    ~ExitTimeObjectMaterialization();
    
    void add(DFG::PromotedLocationDescriptor, const ExitValue&);
    
    DFG::NodeType type() const { return m_type; }
    CodeOrigin origin() const { return m_origin; }
    
    ExitValue get(DFG::PromotedLocationDescriptor) const;
    const Vector<ExitPropertyValue>& properties() const { return m_properties; }
    
    void accountForLocalsOffset(int offset);
    
    void dump(PrintStream& out) const;
    
    void validateReferences(const TrackedReferences&) const;
    
private:
    DFG::NodeType m_type;
    CodeOrigin m_origin;
    Vector<ExitPropertyValue> m_properties;
};

} } // namespace JSC::FTL

#endif // ENABLE(FTL_JIT)
