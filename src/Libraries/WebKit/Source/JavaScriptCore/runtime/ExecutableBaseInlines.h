/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 25, 2025.
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

#include "ExecutableBase.h"
#include "FunctionExecutable.h"
#include "ImplementationVisibility.h"
#include "NativeExecutable.h"
#include "ScriptExecutable.h"
#include "StructureInlines.h"

namespace JSC {

inline Structure* ExecutableBase::createStructure(VM& vm, JSGlobalObject* globalObject, JSValue proto)
{
    return Structure::create(vm, globalObject, proto, TypeInfo(CellType, StructureFlags), info());
}

inline Intrinsic ExecutableBase::intrinsic() const
{
    if (isHostFunction())
        return jsCast<const NativeExecutable*>(this)->intrinsic();
    return jsCast<const ScriptExecutable*>(this)->intrinsic();
}

inline Intrinsic ExecutableBase::intrinsicFor(CodeSpecializationKind kind) const
{
    if (isCall(kind))
        return intrinsic();
    return NoIntrinsic;
}

inline ImplementationVisibility ExecutableBase::implementationVisibility() const
{
    if (isFunctionExecutable())
        return jsCast<const FunctionExecutable*>(this)->implementationVisibility();
    if (isHostFunction())
        return jsCast<const NativeExecutable*>(this)->implementationVisibility();
    return ImplementationVisibility::Public;
}

inline InlineAttribute ExecutableBase::inlineAttribute() const
{
    if (isFunctionExecutable())
        return jsCast<const FunctionExecutable*>(this)->inlineAttribute();
    return InlineAttribute::None;
}

inline bool ExecutableBase::hasJITCodeForCall() const
{
    if (isHostFunction())
        return true;
    return jsCast<const ScriptExecutable*>(this)->hasJITCodeForCall();
}

inline bool ExecutableBase::hasJITCodeForConstruct() const
{
    if (isHostFunction())
        return true;
    return jsCast<const ScriptExecutable*>(this)->hasJITCodeForConstruct();
}

} // namespace JSC
