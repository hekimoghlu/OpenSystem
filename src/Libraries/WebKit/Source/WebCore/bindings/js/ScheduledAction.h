/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 9, 2024.
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

#include <JavaScriptCore/Strong.h>
#include <JavaScriptCore/StrongInlines.h>
#include <memory>
#include <wtf/TZoneMalloc.h>
#include <wtf/text/WTFString.h>

namespace JSC {
class JSGlobalObject;
}

namespace WebCore {

class DOMWrapperWorld;
class Document;
class ScriptExecutionContext;
class WorkerGlobalScope;

class ScheduledAction {
    WTF_MAKE_TZONE_ALLOCATED(ScheduledAction);
public:
    static std::unique_ptr<ScheduledAction> create(DOMWrapperWorld&, JSC::Strong<JSC::JSObject>&&);
    static std::unique_ptr<ScheduledAction> create(DOMWrapperWorld&, String&&);
    ~ScheduledAction();

    void addArguments(FixedVector<JSC::Strong<JSC::Unknown>>&&);

    enum class Type { Code, Function };
    Type type() const;

    StringView code() const { return m_code; }

    void execute(ScriptExecutionContext&);

private:
    ScheduledAction(DOMWrapperWorld&, JSC::Strong<JSC::JSObject>&&);
    ScheduledAction(DOMWrapperWorld&, String&&);

    void executeFunctionInContext(JSC::JSGlobalObject*, JSC::JSValue thisValue, ScriptExecutionContext&);
    void execute(Document&);
    void execute(WorkerGlobalScope&);

    Ref<DOMWrapperWorld> m_isolatedWorld;
    JSC::Strong<JSC::JSObject> m_function;
    FixedVector<JSC::Strong<JSC::Unknown>> m_arguments;
    String m_code;
    JSC::SourceTaintedOrigin m_sourceTaintedOrigin;
};

} // namespace WebCore
