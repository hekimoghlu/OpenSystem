/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 16, 2022.
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
#include "config.h"
#include "PerformanceMark.h"

#include "DOMWrapperWorld.h"
#include "Document.h"
#include "LocalDOMWindow.h"
#include "MessagePort.h"
#include "Performance.h"
#include "PerformanceMarkOptions.h"
#include "PerformanceUserTiming.h"
#include "SerializedScriptValue.h"
#include "WorkerGlobalScope.h"
#include <JavaScriptCore/JSCJSValueInlines.h>

namespace WebCore {

static double performanceNow(ScriptExecutionContext& scriptExecutionContext)
{
    // FIXME: We should consider moving the Performance object to be owned by the
    // the ScriptExecutionContext to avoid this.

    if (RefPtr document = dynamicDowncast<Document>(scriptExecutionContext)) {
        if (auto window = document->domWindow())
            return window->performance().now();
    } else if (RefPtr workerGlobal = dynamicDowncast<WorkerGlobalScope>(scriptExecutionContext))
        return workerGlobal->performance().now();

    return 0;
}

ExceptionOr<Ref<PerformanceMark>> PerformanceMark::create(JSC::JSGlobalObject& globalObject, ScriptExecutionContext& scriptExecutionContext, const String& name, std::optional<PerformanceMarkOptions>&& markOptions)
{
    if (is<Document>(scriptExecutionContext) && PerformanceUserTiming::isRestrictedMarkName(name))
        return Exception { ExceptionCode::SyntaxError };

    double startTime;
    JSC::JSValue detail;
    if (markOptions) {
        if (markOptions->startTime) {
            if (*markOptions->startTime < 0)
                return Exception { ExceptionCode::TypeError };
            startTime = *markOptions->startTime;
        } else
            startTime = performanceNow(scriptExecutionContext);
        
        if (markOptions->detail.isUndefined())
            detail = JSC::jsNull();
        else
            detail = markOptions->detail;
    } else {
        startTime = performanceNow(scriptExecutionContext);
        detail = JSC::jsNull();
    }

    Vector<Ref<MessagePort>> ignoredMessagePorts;
    auto serializedDetail = SerializedScriptValue::create(globalObject, detail, { }, ignoredMessagePorts);
    if (serializedDetail.hasException())
        return serializedDetail.releaseException();

    return adoptRef(*new PerformanceMark(name, startTime, serializedDetail.releaseReturnValue()));
}

PerformanceMark::PerformanceMark(const String& name, double startTime, Ref<SerializedScriptValue>&& serializedDetail)
    : PerformanceEntry(name, startTime, startTime)
    , m_serializedDetail(WTFMove(serializedDetail))
{
}

PerformanceMark::~PerformanceMark() = default;

JSC::JSValue PerformanceMark::detail(JSC::JSGlobalObject& globalObject)
{
    return m_serializedDetail->deserialize(globalObject, &globalObject);
}

}
