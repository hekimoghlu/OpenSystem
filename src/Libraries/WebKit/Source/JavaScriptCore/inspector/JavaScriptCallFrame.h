/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 13, 2025.
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

#include "DebuggerCallFrame.h"
#include "JSCJSValueInlines.h"
#include <wtf/Forward.h>
#include <wtf/RefCounted.h>
#include <wtf/text/TextPosition.h>

namespace Inspector {

class JavaScriptCallFrame : public RefCounted<JavaScriptCallFrame> {
public:
    static Ref<JavaScriptCallFrame> create(Ref<JSC::DebuggerCallFrame>&& debuggerCallFrame)
    {
        return adoptRef(*new JavaScriptCallFrame(WTFMove(debuggerCallFrame)));
    }

    JavaScriptCallFrame* caller();
    JSC::SourceID sourceID() const { return m_debuggerCallFrame->sourceID(); }
    const TextPosition position() const { return m_debuggerCallFrame->position(); }
    int line() const { return m_debuggerCallFrame->line(); }
    int column() const { return m_debuggerCallFrame->column(); }

    String functionName(JSC::VM& vm) const { return m_debuggerCallFrame->functionName(vm); }
    JSC::DebuggerCallFrame::Type type(JSC::VM& vm) const { return m_debuggerCallFrame->type(vm); }
    JSC::DebuggerScope* scopeChain(JSC::VM& vm) const { return m_debuggerCallFrame->scope(vm); }
    bool isTailDeleted() const { return m_debuggerCallFrame->isTailDeleted(); }

    JSC::JSValue thisValue(JSC::VM& vm) const { return m_debuggerCallFrame->thisValue(vm); }
    JSC::JSValue evaluateWithScopeExtension(JSC::VM& vm, const String& script, JSC::JSObject* scopeExtension, NakedPtr<JSC::Exception>& exception) const { return m_debuggerCallFrame->evaluateWithScopeExtension(vm, script, scopeExtension, exception); }

private:
    JavaScriptCallFrame(Ref<JSC::DebuggerCallFrame>&&);

    Ref<JSC::DebuggerCallFrame> m_debuggerCallFrame;
    RefPtr<JavaScriptCallFrame> m_caller;
};

} // namespace Inspector
