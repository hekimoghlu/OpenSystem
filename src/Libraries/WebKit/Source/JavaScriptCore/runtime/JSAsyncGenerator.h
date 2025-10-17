/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 16, 2022.
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

#include "JSGenerator.h"
#include "JSInternalFieldObjectImpl.h"

namespace JSC {

class JSAsyncGenerator final : public JSInternalFieldObjectImpl<8> {
public:
    using Base = JSInternalFieldObjectImpl<8>;

    // JSAsyncGenerator has one inline storage slot, which is pointing internalField(0).
    static size_t allocationSize(Checked<size_t> inlineCapacity)
    {
        ASSERT_UNUSED(inlineCapacity, inlineCapacity == 0U);
        return sizeof(JSAsyncGenerator);
    }

    template<typename CellType, SubspaceAccess mode>
    static GCClient::IsoSubspace* subspaceFor(VM& vm)
    {
        return vm.asyncGeneratorSpace<mode>();
    }

    enum class AsyncGeneratorState : int32_t {
        Completed = -1,
        Executing = -2,
        SuspendedStart = -3,
        SuspendedYield = -4,
        AwaitingReturn = -5,
    };
    static_assert(static_cast<int32_t>(AsyncGeneratorState::Completed) == static_cast<int32_t>(JSGenerator::State::Completed));
    static_assert(static_cast<int32_t>(AsyncGeneratorState::Executing) == static_cast<int32_t>(JSGenerator::State::Executing));

    enum class AsyncGeneratorSuspendReason : int32_t {
        None = 0,
        Yield = -1,
        Await = -2
    };

    enum class Field : uint32_t {
        // FIXME: JSAsyncGenerator should support PolyProto, since generator tends to be created with poly proto mode.
        // We reserve the first internal field for PolyProto property. This offset is identical to JSFinalObject's first inline storage slot which will be used for PolyProto.
        PolyProto = 0,
        State,
        Next,
        This,
        Frame,
        SuspendReason,
        QueueFirst,
        QueueLast,
    };
    static_assert(numberOfInternalFields == 8);
    static std::array<JSValue, numberOfInternalFields> initialValues()
    {
        return { {
            jsNull(),
            jsNumber(static_cast<int32_t>(AsyncGeneratorState::SuspendedStart)),
            jsUndefined(),
            jsUndefined(),
            jsUndefined(),
            jsNumber(static_cast<int32_t>(AsyncGeneratorSuspendReason::None)),
            jsNull(),
            jsNull(),
        } };
    }

    static JSAsyncGenerator* create(VM&, Structure*);
    static Structure* createStructure(VM&, JSGlobalObject*, JSValue);

    DECLARE_EXPORT_INFO;

    DECLARE_VISIT_CHILDREN;

private:
    JSAsyncGenerator(VM&, Structure*);
    void finishCreation(VM&);
};

} // namespace JSC
