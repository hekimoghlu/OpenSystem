/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 28, 2023.
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

#include "DeferGC.h"
#include "DisallowVMEntry.h"
#include "VM.h"

namespace JSC {

class VM;
class JSObject;

#if ASSERT_ENABLED

class ObjectInitializationScope {
public:
    JS_EXPORT_PRIVATE ObjectInitializationScope(VM&);
    JS_EXPORT_PRIVATE ~ObjectInitializationScope();

    VM& vm() const { return m_vm; }
    void notifyAllocated(JSObject*);
    void notifyInitialized(JSObject*);

private:
    void verifyPropertiesAreInitialized(JSObject*);

    VM& m_vm;
    std::optional<DisallowGC> m_disallowGC;
    std::optional<DisallowVMEntry> m_disallowVMEntry;
    JSObject* m_object { nullptr };
};

#else // not ASSERT_ENABLED

class ObjectInitializationScope {
public:
    ALWAYS_INLINE ObjectInitializationScope(VM& vm)
        : m_vm(vm)
    { }
    ALWAYS_INLINE ~ObjectInitializationScope()
    {
        m_vm.mutatorFence();
    }

    ALWAYS_INLINE VM& vm() const { return m_vm; }
    ALWAYS_INLINE void notifyAllocated(JSObject*) { }
    ALWAYS_INLINE void notifyInitialized(JSObject*) { }

private:
    VM& m_vm;
};

#endif // ASSERT_ENABLED

} // namespace JSC
