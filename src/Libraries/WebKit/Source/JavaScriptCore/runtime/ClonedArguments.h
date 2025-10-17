/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 7, 2024.
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

#include "ArgumentsMode.h"
#include "JSObject.h"

namespace JSC {

// This is an Arguments-class object that we create when you do function.arguments, or you say
// "arguments" inside a function in strict mode. It behaves almpst entirely like an ordinary
// JavaScript object. All of the arguments values are simply copied from the stack (possibly via
// some sophisticated ValueRecovery's if an optimizing compiler is in play) and the appropriate
// properties of the object are populated. The only reason why we need a special class is to make
// the object claim to be "Arguments" from a toString standpoint, and to avoid materializing the
// caller/callee/@@iterator properties unless someone asks for them.

static constexpr PropertyOffset clonedArgumentsLengthPropertyOffset = firstOutOfLineOffset;

class ClonedArguments final : public JSNonFinalObject {
public:
    using Base = JSNonFinalObject;
    static constexpr unsigned StructureFlags = Base::StructureFlags | OverridesGetOwnPropertySlot | OverridesGetOwnSpecialPropertyNames | OverridesPut;

    template<typename CellType, SubspaceAccess mode>
    static GCClient::IsoSubspace* subspaceFor(VM& vm)
    {
        static_assert(CellType::needsDestruction == DoesNotNeedDestruction);
        return &vm.clonedArgumentsSpace();
    }

    uint64_t length(JSGlobalObject* globalObject) const
    {
        VM& vm = getVM(globalObject);
        auto scope = DECLARE_THROW_SCOPE(vm);

        JSValue lengthValue;
        if (LIKELY(!structure()->didTransition())) {
            lengthValue = getDirect(clonedArgumentsLengthPropertyOffset);
            if (LIKELY(lengthValue.isInt32()))
                return std::max(lengthValue.asInt32(), 0);
        } else {
            lengthValue = get(globalObject, vm.propertyNames->length);
            RETURN_IF_EXCEPTION(scope, 0);
        }
        RELEASE_AND_RETURN(scope, lengthValue.toLength(globalObject));
    }

    void copyToArguments(JSGlobalObject*, JSValue* firstElementDest, unsigned offset, unsigned length);

    JS_EXPORT_PRIVATE bool isIteratorProtocolFastAndNonObservable();
    
private:
    ClonedArguments(VM&, Structure*, Butterfly*);

public:
    static ClonedArguments* createEmpty(VM&, JSGlobalObject* nullOrGlobalObjectForOOM, Structure*, JSFunction* callee, unsigned length, Butterfly*);
    static ClonedArguments* createWithInlineFrame(JSGlobalObject*, CallFrame* targetFrame, InlineCallFrame*, ArgumentsMode);
    static ClonedArguments* createWithMachineFrame(JSGlobalObject*, CallFrame* targetFrame, ArgumentsMode);
    static ClonedArguments* createByCopyingFrom(JSGlobalObject*, Structure*, Register* argumentsStart, unsigned length, JSFunction* callee, Butterfly*);
    
    static Structure* createStructure(VM&, JSGlobalObject*, JSValue prototype);
    static Structure* createSlowPutStructure(VM&, JSGlobalObject*, JSValue prototype);

    static constexpr ptrdiff_t offsetOfCallee()
    {
        return OBJECT_OFFSETOF(ClonedArguments, m_callee);
    }

    static size_t allocationSize(Checked<size_t> inlineCapacity)
    {
        ASSERT_UNUSED(inlineCapacity, !inlineCapacity);
        return sizeof(ClonedArguments);
    }

    DECLARE_VISIT_CHILDREN;

    DECLARE_INFO;

private:
    static Structure* createStructure(VM&, JSGlobalObject*, JSValue prototype, IndexingType);

    static bool getOwnPropertySlot(JSObject*, JSGlobalObject*, PropertyName, PropertySlot&);
    static void getOwnSpecialPropertyNames(JSObject*, JSGlobalObject*, PropertyNameArray&, DontEnumPropertiesMode);
    static bool put(JSCell*, JSGlobalObject*, PropertyName, JSValue, PutPropertySlot&);
    static bool deleteProperty(JSCell*, JSGlobalObject*, PropertyName, DeletePropertySlot&);
    static bool defineOwnProperty(JSObject*, JSGlobalObject*, PropertyName, const PropertyDescriptor&, bool shouldThrow);
    
    bool specialsMaterialized() const { return !m_callee; }
    void materializeSpecials(JSGlobalObject*);
    void materializeSpecialsIfNecessary(JSGlobalObject*);
    
    WriteBarrier<JSFunction> m_callee; // Set to nullptr when we materialize all of our special properties.
};

} // namespace JSC
