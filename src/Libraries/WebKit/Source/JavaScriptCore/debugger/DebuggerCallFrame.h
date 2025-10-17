/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 4, 2024.
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

#include "CallFrame.h"
#include "DebuggerPrimitives.h"
#include "ShadowChicken.h"
#include "Strong.h"
#include <wtf/NakedPtr.h>
#include <wtf/RefCounted.h>
#include <wtf/text/TextPosition.h>

namespace JSC {

class DebuggerScope;
class Exception;

class DebuggerCallFrame : public RefCounted<DebuggerCallFrame> {
public:
    enum Type { ProgramType, FunctionType };

    static Ref<DebuggerCallFrame> create(VM&, CallFrame*);

    JS_EXPORT_PRIVATE RefPtr<DebuggerCallFrame> callerFrame();
    JSGlobalObject* globalObject(VM&);
    JS_EXPORT_PRIVATE SourceID sourceID() const;

    // line and column are in base 0 e.g. the first line is line 0.
    int line() const { return m_position.m_line.zeroBasedInt(); }
    int column() const { return m_position.m_column.zeroBasedInt(); }
    JS_EXPORT_PRIVATE const TextPosition& position() const { return m_position; }

    JS_EXPORT_PRIVATE DebuggerScope* scope(VM&);
    JS_EXPORT_PRIVATE String functionName(VM&) const;
    JS_EXPORT_PRIVATE Type type(VM&) const;
    JS_EXPORT_PRIVATE JSValue thisValue(VM&) const;

    JSValue evaluateWithScopeExtension(VM&, const String&, JSObject* scopeExtensionObject, NakedPtr<Exception>&);

    bool isValid() const { return !!m_validMachineFrame || isTailDeleted(); }
    JS_EXPORT_PRIVATE void invalidate();

    // The following are only public for the Debugger's use only. They will be
    // made private soon. Other clients should not use these.

    JS_EXPORT_PRIVATE TextPosition currentPosition(VM&);
    JS_EXPORT_PRIVATE static TextPosition positionForCallFrame(VM&, CallFrame*);
    JS_EXPORT_PRIVATE static SourceID sourceIDForCallFrame(CallFrame*);

    bool isTailDeleted() const { return m_shadowChickenFrame.isTailDeleted; }

private:
    DebuggerCallFrame(VM&, CallFrame*, const ShadowChicken::Frame&);

    CallFrame* m_validMachineFrame;
    RefPtr<DebuggerCallFrame> m_caller;
    TextPosition m_position;
    // The DebuggerPausedScope is responsible for calling invalidate() which,
    // in turn, will clear this strong ref.
    Strong<DebuggerScope> m_scope;
    ShadowChicken::Frame m_shadowChickenFrame;
};

} // namespace JSC
