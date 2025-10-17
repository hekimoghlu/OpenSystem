/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 2, 2025.
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
#ifndef WebScriptDebugger_h
#define WebScriptDebugger_h

#include <JavaScriptCore/Debugger.h>
#include <JavaScriptCore/Strong.h>

#include <wtf/Forward.h>
#include <wtf/RetainPtr.h>

namespace JSC {
    class CallFrame;
    class DebuggerCallFrame;
    class JSGlobalObject;
    class JSObject;
    class ArgList;
}

@class WebScriptCallFrame;

class WebScriptDebugger final : public JSC::Debugger {
public:
    WebScriptDebugger(JSC::JSGlobalObject*);

    JSC::JSGlobalObject* globalObject() const { return m_globalObject.get(); }

private:
    void sourceParsed(JSC::JSGlobalObject*, JSC::SourceProvider*, int errorLine, const WTF::String& errorMsg) override;
    void handlePause(JSC::JSGlobalObject*, JSC::Debugger::ReasonForPause) override;

    bool m_callingDelegate;

    JSC::Strong<JSC::JSGlobalObject> m_globalObject;
};

#endif // WebScriptDebugger_h
