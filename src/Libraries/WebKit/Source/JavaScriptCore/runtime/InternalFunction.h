/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 5, 2022.
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

#include "CodeSpecializationKind.h"
#include "JSDestructibleObject.h"

namespace JSC {

class FunctionPrototype;

class InternalFunction : public JSNonFinalObject {
    friend class JIT;
    friend class LLIntOffsetsExtractor;
public:
    using Base = JSNonFinalObject;
    static constexpr unsigned StructureFlags = Base::StructureFlags | ImplementsHasInstance | ImplementsDefaultHasInstance | OverridesGetCallData;

    template<typename CellType, SubspaceAccess>
    static GCClient::IsoSubspace* subspaceFor(VM& vm)
    {
        static_assert(sizeof(CellType) == sizeof(InternalFunction), "InternalFunction subclasses that add fields need to override subspaceFor<>()");
        return &vm.internalFunctionSpace();
    }

    DECLARE_EXPORT_INFO;

    DECLARE_VISIT_CHILDREN_WITH_MODIFIER(JS_EXPORT_PRIVATE);

    JS_EXPORT_PRIVATE String name();
    String displayName(VM&);
    String calculatedDisplayName(VM&);

    inline static Structure* createStructure(VM&, JSGlobalObject*, JSValue);

    JS_EXPORT_PRIVATE static Structure* createSubclassStructure(JSGlobalObject*, JSObject* newTarget, Structure*);
    JS_EXPORT_PRIVATE static InternalFunction* createFunctionThatMasqueradesAsUndefined(VM&, JSGlobalObject*, unsigned length, const String& name, NativeFunction);

    TaggedNativeFunction nativeFunctionFor(CodeSpecializationKind kind)
    {
        if (kind == CodeForCall)
            return m_functionForCall;
        ASSERT(kind == CodeForConstruct);
        return m_functionForConstruct;
    }

    static constexpr ptrdiff_t offsetOfNativeFunctionFor(CodeSpecializationKind kind)
    {
        if (kind == CodeForCall)
            return OBJECT_OFFSETOF(InternalFunction, m_functionForCall);
        ASSERT(kind == CodeForConstruct);
        return OBJECT_OFFSETOF(InternalFunction, m_functionForConstruct);
    }

    static constexpr ptrdiff_t offsetOfGlobalObject()
    {
        return OBJECT_OFFSETOF(InternalFunction, m_globalObject);
    }

    JSGlobalObject* globalObject() const { return m_globalObject.get(); }

protected:
    JS_EXPORT_PRIVATE InternalFunction(VM&, Structure*, NativeFunction functionForCall, NativeFunction functionForConstruct = nullptr);

    enum class PropertyAdditionMode { WithStructureTransition, WithoutStructureTransition };
    JS_EXPORT_PRIVATE void finishCreation(VM&, unsigned length, const String& name, PropertyAdditionMode = PropertyAdditionMode::WithStructureTransition);
    DECLARE_DEFAULT_FINISH_CREATION;

    JS_EXPORT_PRIVATE static CallData getConstructData(JSCell*);
    JS_EXPORT_PRIVATE static CallData getCallData(JSCell*);

    TaggedNativeFunction m_functionForCall;
    TaggedNativeFunction m_functionForConstruct;
    WriteBarrier<JSString> m_originalName;
    WriteBarrier<JSGlobalObject> m_globalObject;
};

JS_EXPORT_PRIVATE JSGlobalObject* getFunctionRealm(JSGlobalObject*, JSObject*);

#define JSC_GET_DERIVED_STRUCTURE(vm, structureMemberFunctionName, newTarget, constructor) \
    ((newTarget) == (constructor) \
        ? globalObject->structureMemberFunctionName() \
        : ([&]() -> Structure* { \
            auto scope = DECLARE_THROW_SCOPE((vm)); \
            auto* functionGlobalObject = getFunctionRealm(globalObject, (newTarget)); \
            RETURN_IF_EXCEPTION(scope, nullptr); \
            RELEASE_AND_RETURN(scope, InternalFunction::createSubclassStructure(globalObject, (newTarget), functionGlobalObject->structureMemberFunctionName())); \
        }()))

} // namespace JSC
