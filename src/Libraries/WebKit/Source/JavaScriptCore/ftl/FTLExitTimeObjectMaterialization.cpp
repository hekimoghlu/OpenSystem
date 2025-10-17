/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 20, 2021.
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
#include "config.h"
#include "FTLExitTimeObjectMaterialization.h"

#if ENABLE(FTL_JIT)

#include "DFGGraph.h"

namespace JSC { namespace FTL {

using namespace JSC::DFG;

ExitTimeObjectMaterialization::ExitTimeObjectMaterialization(NodeType type, CodeOrigin codeOrigin)
    : m_type(type)
    , m_origin(codeOrigin)
{
}

ExitTimeObjectMaterialization::~ExitTimeObjectMaterialization() = default;

void ExitTimeObjectMaterialization::add(
    PromotedLocationDescriptor location, const ExitValue& value)
{
    m_properties.append(ExitPropertyValue(location, value));
}

ExitValue ExitTimeObjectMaterialization::get(PromotedLocationDescriptor location) const
{
    for (ExitPropertyValue value : m_properties) {
        if (value.location() == location)
            return value.value();
    }
    return ExitValue();
}

void ExitTimeObjectMaterialization::accountForLocalsOffset(int offset)
{
    for (ExitPropertyValue& property : m_properties)
        property = property.withLocalsOffset(offset);
}

void ExitTimeObjectMaterialization::dump(PrintStream& out) const
{
    out.print(RawPointer(this), ":", Graph::opName(m_type), "(", listDump(m_properties), ")");
}

void ExitTimeObjectMaterialization::validateReferences(const TrackedReferences& trackedReferences) const
{
    for (ExitPropertyValue value : m_properties)
        value.validateReferences(trackedReferences);
}

} } // namespace JSC::FTL

#endif // ENABLE(FTL_JIT)

