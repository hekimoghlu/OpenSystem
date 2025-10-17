/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 1, 2024.
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

#include "InspectorInstrumentation.h"
#include "JSExecState.h"
#include <JavaScriptCore/FunctionExecutable.h>

namespace WebCore {

inline void JSExecState::instrumentFunction(ScriptExecutionContext* context, const JSC::CallData& callData)
{
    if (!InspectorInstrumentation::timelineAgentTracking(context))
        return;

    String resourceName;
    int lineNumber = 1;
    int columnNumber = 1;
    if (callData.type == JSC::CallData::Type::JS) {
        resourceName = callData.js.functionExecutable->sourceURL();
        lineNumber = callData.js.functionExecutable->firstLine();
        columnNumber = callData.js.functionExecutable->startColumn();
    } else
        resourceName = "undefined"_s;
    InspectorInstrumentation::willCallFunction(context, resourceName, lineNumber, columnNumber);
}

} // namespace WebCore
