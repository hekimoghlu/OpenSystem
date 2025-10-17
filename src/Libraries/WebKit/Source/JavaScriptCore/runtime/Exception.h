/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 23, 2023.
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

#include "JSDestructibleObject.h"
#include "StackFrame.h"
#include <wtf/Vector.h>

namespace JSC {
    
class Exception final : public JSCell {
public:
    using Base = JSCell;
    static constexpr unsigned StructureFlags = Base::StructureFlags | StructureIsImmortal;
    static constexpr DestructionMode needsDestruction = NeedsDestruction;

    template<typename CellType, SubspaceAccess mode>
    static GCClient::IsoSubspace* subspaceFor(VM& vm)
    {
        return &vm.exceptionSpace();
    }

    enum StackCaptureAction {
        CaptureStack,
        DoNotCaptureStack
    };
    JS_EXPORT_PRIVATE static Exception* create(VM&, JSValue thrownValue, StackCaptureAction = CaptureStack);

    static Structure* createStructure(VM&, JSGlobalObject*, JSValue prototype);

    DECLARE_VISIT_CHILDREN;

    DECLARE_EXPORT_INFO;

    static constexpr ptrdiff_t valueOffset()
    {
        return OBJECT_OFFSETOF(Exception, m_value);
    }

    JSValue value() const { return m_value.get(); }
    const Vector<StackFrame>& stack() const { return m_stack; }

    bool didNotifyInspectorOfThrow() const { return m_didNotifyInspectorOfThrow; }
    void setDidNotifyInspectorOfThrow() { m_didNotifyInspectorOfThrow = true; }

#if ENABLE(WEBASSEMBLY)
    void wrapValueForJSTag(JSGlobalObject*);
#endif

    ~Exception();

private:
    Exception(VM&, JSValue thrownValue);
    void finishCreation(VM&, StackCaptureAction);
    static void destroy(JSCell*);

    WriteBarrier<Unknown> m_value;
    Vector<StackFrame> m_stack;
    bool m_didNotifyInspectorOfThrow { false };

    friend class LLIntOffsetsExtractor;
};

} // namespace JSC
