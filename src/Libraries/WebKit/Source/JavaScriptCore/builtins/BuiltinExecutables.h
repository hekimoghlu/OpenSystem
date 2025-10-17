/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 3, 2022.
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

#include "ExecutableInfo.h"
#include "JSCBuiltins.h"
#include "ParserModes.h"
#include "SourceCode.h"
#include "Weak.h"
#include "WeakHandleOwner.h"
#include <wtf/TZoneMalloc.h>

namespace JSC {

class UnlinkedFunctionExecutable;
class Identifier;
class VM;

#define BUILTIN_NAME_ONLY(name, functionName, overriddenName, length) name,
enum class BuiltinCodeIndex {
    JSC_FOREACH_BUILTIN_CODE(BUILTIN_NAME_ONLY)
    NumberOfBuiltinCodes
};
#undef BUILTIN_NAME_ONLY

class BuiltinExecutables {
    WTF_MAKE_TZONE_ALLOCATED(BuiltinExecutables);
public:
    explicit BuiltinExecutables(VM&);

#define EXPOSE_BUILTIN_EXECUTABLES(name, functionName, overriddenName, length) \
UnlinkedFunctionExecutable* name##Executable(); \
SourceCode name##Source();
    
    JSC_FOREACH_BUILTIN_CODE(EXPOSE_BUILTIN_EXECUTABLES)
#undef EXPOSE_BUILTIN_EXECUTABLES

    static SourceCode defaultConstructorSourceCode(ConstructorKind);
    UnlinkedFunctionExecutable* createDefaultConstructor(ConstructorKind, const Identifier& name, NeedsClassFieldInitializer, PrivateBrandRequirement);

    static UnlinkedFunctionExecutable* createExecutable(VM&, const SourceCode&, const Identifier&, ImplementationVisibility, ConstructorKind, ConstructAbility, InlineAttribute, NeedsClassFieldInitializer, PrivateBrandRequirement = PrivateBrandRequirement::None);

    DECLARE_VISIT_AGGREGATE;

    void clear();

private:
    VM& m_vm;

    UnlinkedFunctionExecutable* createBuiltinExecutable(const SourceCode&, const Identifier&, ImplementationVisibility, ConstructorKind, ConstructAbility, InlineAttribute);

    Ref<StringSourceProvider> m_combinedSourceProvider;
    UnlinkedFunctionExecutable* m_unlinkedExecutables[static_cast<unsigned>(BuiltinCodeIndex::NumberOfBuiltinCodes)] { };
};

}
