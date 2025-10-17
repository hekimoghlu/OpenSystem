/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 19, 2024.
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

#include "ArrayBuffer.h"
#include "JSObject.h"

namespace JSC {

class JSArrayBuffer final : public JSNonFinalObject {
public:
    using Base = JSNonFinalObject;
    static constexpr unsigned StructureFlags = Base::StructureFlags;

    template<typename CellType, SubspaceAccess mode>
    static GCClient::IsoSubspace* subspaceFor(VM& vm)
    {
        return vm.arrayBufferSpace<mode>();
    }

    // This function will register the new wrapper with the vm's TypedArrayController.
    JS_EXPORT_PRIVATE static JSArrayBuffer* create(VM&, Structure*, RefPtr<ArrayBuffer>&&);

    ArrayBuffer* impl() const { return m_impl; }
    
    static Structure* createStructure(VM&, JSGlobalObject*, JSValue prototype);

    JS_EXPORT_PRIVATE bool isShared() const;
    ArrayBufferSharingMode sharingMode() const;
    bool isResizableOrGrowableShared() const { return m_impl->isResizableOrGrowableShared(); }
    
    DECLARE_EXPORT_INFO;
    
    // This is the default DOM unwrapping. It calls toUnsharedArrayBuffer().
    static ArrayBuffer* toWrapped(VM&, JSValue);
    static ArrayBuffer* toWrappedAllowShared(VM&, JSValue);
    
private:
    JSArrayBuffer(VM&, Structure*, RefPtr<ArrayBuffer>&&);
    void finishCreation(VM&, JSGlobalObject*);

    static size_t estimatedSize(JSCell*, VM&);

    ArrayBuffer* m_impl;
};

inline ArrayBuffer* toPossiblySharedArrayBuffer(VM&, JSValue value)
{
    JSArrayBuffer* wrapper = jsDynamicCast<JSArrayBuffer*>(value);
    if (!wrapper)
        return nullptr;
    return wrapper->impl();
}

inline ArrayBuffer* toUnsharedArrayBuffer(VM& vm, JSValue value)
{
    ArrayBuffer* result = toPossiblySharedArrayBuffer(vm, value);
    if (!result || result->isShared())
        return nullptr;
    return result;
}

inline ArrayBuffer* JSArrayBuffer::toWrapped(VM& vm, JSValue value)
{
    auto result = toUnsharedArrayBuffer(vm, value);
    if (!result || result->isResizableOrGrowableShared())
        return nullptr;
    return result;
}

inline ArrayBuffer* JSArrayBuffer::toWrappedAllowShared(VM& vm, JSValue value)
{
    auto result = toPossiblySharedArrayBuffer(vm, value);
    if (!result || result->isResizableOrGrowableShared())
        return nullptr;
    return result;
}

} // namespace JSC
