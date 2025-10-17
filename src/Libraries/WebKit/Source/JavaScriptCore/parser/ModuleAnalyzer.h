/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 10, 2023.
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

#include "ErrorType.h"
#include "Nodes.h"

namespace JSC {

class JSModuleRecord;
class SourceCode;
class ScriptFetchParameters;

class ModuleAnalyzer {
    WTF_MAKE_NONCOPYABLE(ModuleAnalyzer);
    WTF_FORBID_HEAP_ALLOCATION;
public:
    ModuleAnalyzer(JSGlobalObject*, const Identifier& moduleKey, const SourceCode&, const VariableEnvironment& declaredVariables, const VariableEnvironment& lexicalVariables, CodeFeatures);

    Expected<JSModuleRecord*, std::tuple<ErrorType, String>> analyze(ModuleProgramNode&);

    VM& vm() { return m_vm; }

    JSModuleRecord* moduleRecord() { return m_moduleRecord; }

    void appendRequestedModule(const Identifier&, RefPtr<ScriptFetchParameters>&&);

    void fail(std::tuple<ErrorType, String>&& errorMessage) { m_errorMessage = errorMessage; }

private:
    void exportVariable(ModuleProgramNode&, const RefPtr<UniquedStringImpl>&, const VariableEnvironmentEntry&);

    VM& m_vm;
    JSModuleRecord* m_moduleRecord;
    IdentifierSet m_requestedModules;
    std::tuple<ErrorType, String> m_errorMessage;
};

} // namespace JSC
