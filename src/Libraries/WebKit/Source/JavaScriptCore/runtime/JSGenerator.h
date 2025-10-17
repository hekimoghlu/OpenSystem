/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 22, 2022.
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

#include "JSInternalFieldObjectImpl.h"

namespace JSC {

class JSGenerator final : public JSInternalFieldObjectImpl<6> {
public:
    using Base = JSInternalFieldObjectImpl<6>;

    // JSGenerator has one inline storage slot, which is pointing internalField(0).
    static size_t allocationSize(Checked<size_t> inlineCapacity)
    {
        ASSERT_UNUSED(inlineCapacity, inlineCapacity == 0U);
        return sizeof(JSGenerator);
    }

    template<typename CellType, SubspaceAccess mode>
    static GCClient::IsoSubspace* subspaceFor(VM& vm)
    {
        return vm.generatorSpace<mode>();
    }

    enum class ResumeMode : int32_t {
        NormalMode = 0,
        ReturnMode = 1,
        ThrowMode = 2
    };

    enum class State : int32_t {
        Completed = -1,
        Executing = -2,
        Init = 0,
    };

    // [this], @generator, @generatorState, @generatorValue, @generatorResumeMode, @generatorFrame.
    enum class Argument : int32_t {
        ThisValue = 0,
        Generator = 1,
        State = 2,
        Value = 3,
        ResumeMode = 4,
        Frame = 5,
        NumberOfArguments = Frame,
    };

    enum class Field : uint32_t {
        // FIXME: JSGenerator should support PolyProto, since generator tends to be created with poly proto mode.
        // We reserve the first internal field for PolyProto property. This offset is identical to JSFinalObject's first inline storage slot which will be used for PolyProto.
        PolyProto = 0,
        State,
        Next,
        This,
        Frame,
        Context,
    };
    static_assert(numberOfInternalFields == 6);
    static std::array<JSValue, numberOfInternalFields> initialValues()
    {
        return { {
            jsNull(),
            jsNumber(static_cast<int32_t>(State::Init)),
            jsUndefined(),
            jsUndefined(),
            jsUndefined(),
            jsUndefined(),
        } };
    }

    static JSGenerator* create(VM&, Structure*);
    static Structure* createStructure(VM&, JSGlobalObject*, JSValue);

    DECLARE_EXPORT_INFO;

    DECLARE_VISIT_CHILDREN;

private:
    JSGenerator(VM&, Structure*);
    void finishCreation(VM&);
};

OVERLOAD_RELATIONAL_OPERATORS_FOR_ENUM_CLASS_WITH_INTEGRALS(JSGenerator::Argument);

} // namespace JSC
