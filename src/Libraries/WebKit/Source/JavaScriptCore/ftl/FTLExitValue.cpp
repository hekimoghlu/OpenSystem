/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 8, 2022.
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
#include "FTLExitValue.h"

#if ENABLE(FTL_JIT)

#include "JSCJSValueInlines.h"
#include "TrackedReferences.h"

namespace JSC { namespace FTL {

ExitValue ExitValue::materializeNewObject(ExitTimeObjectMaterialization* data)
{
    ExitValue result;
    result.m_kind = ExitValueMaterializeNewObject;
    UnionType u;
    u.newObjectMaterializationData = data;
    result.m_value = WTFMove(u);
    return result;
}

ExitValue ExitValue::withLocalsOffset(int offset) const
{
    if (!isInJSStackSomehow())
        return *this;
    if (!virtualRegister().isLocal())
        return *this;
    return withVirtualRegister(virtualRegister() + offset);
}

DataFormat ExitValue::dataFormat() const
{
    switch (kind()) {
    case InvalidExitValue:
        RELEASE_ASSERT_NOT_REACHED();
        return DataFormatNone;
            
    case ExitValueDead:
    case ExitValueConstant:
    case ExitValueInJSStack:
    case ExitValueMaterializeNewObject:
        return DataFormatJS;
            
    case ExitValueArgument:
        return exitArgument().format();
            
    case ExitValueInJSStackAsInt32:
        return DataFormatInt32;
            
    case ExitValueInJSStackAsInt52:
        return DataFormatInt52;
            
    case ExitValueInJSStackAsDouble:
        return DataFormatDouble;
    }
        
    RELEASE_ASSERT_NOT_REACHED();
}

void ExitValue::dumpInContext(PrintStream& out, DumpContext* context) const
{
    switch (kind()) {
    case InvalidExitValue:
        out.print("Invalid");
        return;
    case ExitValueDead:
        out.print("Dead");
        return;
    case ExitValueArgument:
        out.print("Argument(", exitArgument(), ")");
        return;
    case ExitValueConstant:
        out.print("Constant(", inContext(constant(), context), ")");
        return;
    case ExitValueInJSStack:
        out.print("InJSStack:", virtualRegister());
        return;
    case ExitValueInJSStackAsInt32:
        out.print("InJSStackAsInt32:", virtualRegister());
        return;
    case ExitValueInJSStackAsInt52:
        out.print("InJSStackAsInt52:", virtualRegister());
        return;
    case ExitValueInJSStackAsDouble:
        out.print("InJSStackAsDouble:", virtualRegister());
        return;
    case ExitValueMaterializeNewObject:
        out.print("Materialize(", WTF::RawPointer(objectMaterialization()), ")");
        return;
    }
    
    RELEASE_ASSERT_NOT_REACHED();
}

void ExitValue::dump(PrintStream& out) const
{
    dumpInContext(out, nullptr);
}

void ExitValue::validateReferences(const TrackedReferences& trackedReferences) const
{
    if (isConstant())
        trackedReferences.check(constant());
}

} } // namespace JSC::FTL

#endif // ENABLE(FTL_JIT)

