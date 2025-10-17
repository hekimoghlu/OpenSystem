/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 26, 2022.
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

class JSPromiseConstructor;
class JSPromise : public JSInternalFieldObjectImpl<2> {
public:
    using Base = JSInternalFieldObjectImpl<2>;

    template<typename CellType, SubspaceAccess mode>
    static GCClient::IsoSubspace* subspaceFor(VM& vm)
    {
        return &vm.promiseSpace();
    }

    JS_EXPORT_PRIVATE static JSPromise* create(VM&, Structure*);
    static JSPromise* createWithInitialValues(VM&, Structure*);
    static Structure* createStructure(VM&, JSGlobalObject*, JSValue);

    DECLARE_EXPORT_INFO;

    enum class Status : unsigned {
        Pending = 0, // Making this as 0, so that, we can change the status from Pending to others without masking.
        Fulfilled = 1,
        Rejected = 2,
    };
    static constexpr uint32_t isHandledFlag = 4;
    static constexpr uint32_t isFirstResolvingFunctionCalledFlag = 8;
    static constexpr uint32_t stateMask = 0b11;

    enum class Field : unsigned {
        Flags = 0,
        ReactionsOrResult = 1,
    };
    static_assert(numberOfInternalFields == 2);

    static std::array<JSValue, numberOfInternalFields> initialValues()
    {
        return { {
            jsNumber(static_cast<unsigned>(Status::Pending)),
            jsUndefined(),
        } };
    }

    const WriteBarrier<Unknown>& internalField(Field field) const { return Base::internalField(static_cast<uint32_t>(field)); }
    WriteBarrier<Unknown>& internalField(Field field) { return Base::internalField(static_cast<uint32_t>(field)); }

    JS_EXPORT_PRIVATE Status status(VM&) const;
    JS_EXPORT_PRIVATE JSValue result(VM&) const;
    JS_EXPORT_PRIVATE bool isHandled(VM&) const;

    JS_EXPORT_PRIVATE static JSPromise* resolvedPromise(JSGlobalObject*, JSValue);
    JS_EXPORT_PRIVATE static JSPromise* rejectedPromise(JSGlobalObject*, JSValue);

    JS_EXPORT_PRIVATE void resolve(JSGlobalObject*, JSValue);
    JS_EXPORT_PRIVATE void reject(JSGlobalObject*, JSValue);
    JS_EXPORT_PRIVATE void rejectAsHandled(JSGlobalObject*, JSValue);
    JS_EXPORT_PRIVATE void reject(JSGlobalObject*, Exception*);
    JS_EXPORT_PRIVATE void rejectAsHandled(JSGlobalObject*, Exception*);
    JS_EXPORT_PRIVATE void markAsHandled(JSGlobalObject*);
    JS_EXPORT_PRIVATE void performPromiseThen(JSGlobalObject*, JSFunction*, JSFunction*, JSValue);

    JS_EXPORT_PRIVATE JSPromise* rejectWithCaughtException(JSGlobalObject*, ThrowScope&);

    struct DeferredData {
        WTF_FORBID_HEAP_ALLOCATION;
    public:
        JSPromise* promise { nullptr };
        JSFunction* resolve { nullptr };
        JSFunction* reject { nullptr };
    };
    static DeferredData createDeferredData(JSGlobalObject*, JSPromiseConstructor*);
    JS_EXPORT_PRIVATE static JSValue createNewPromiseCapability(JSGlobalObject*, JSPromiseConstructor*);
    JS_EXPORT_PRIVATE static DeferredData convertCapabilityToDeferredData(JSGlobalObject*, JSValue);

    DECLARE_VISIT_CHILDREN;

protected:
    JSPromise(VM&, Structure*);
    void finishCreation(VM&);

    uint32_t flags() const;
};

} // namespace JSC
