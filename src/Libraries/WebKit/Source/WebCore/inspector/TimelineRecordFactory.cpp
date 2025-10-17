/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 23, 2021.
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
#include "TimelineRecordFactory.h"

#include "Event.h"
#include "FloatQuad.h"
#include "JSExecState.h"
#include <JavaScriptCore/InspectorProtocolObjects.h>
#include <JavaScriptCore/ScriptCallStack.h>
#include <JavaScriptCore/ScriptCallStackFactory.h>

namespace WebCore {

using namespace Inspector;

Ref<JSON::Object> TimelineRecordFactory::createGenericRecord(double startTime, int maxCallStackDepth)
{
    Ref<JSON::Object> record = JSON::Object::create();
    record->setDouble("startTime"_s, startTime);

    if (maxCallStackDepth) {
        Ref<ScriptCallStack> stackTrace = createScriptCallStack(JSExecState::currentState(), maxCallStackDepth);
        if (stackTrace->size())
            record->setValue("stackTrace"_s, stackTrace->buildInspectorObject());
    }
    return record;
}

Ref<JSON::Object> TimelineRecordFactory::createRenderingFrameData(const String& name)
{
    Ref<JSON::Object> data = JSON::Object::create();
    data->setString("name"_s, name);
    return data;
}

Ref<JSON::Object> TimelineRecordFactory::createFunctionCallData(const String& scriptName, int scriptLine, int scriptColumn)
{
    Ref<JSON::Object> data = JSON::Object::create();
    data->setString("scriptName"_s, scriptName);
    data->setInteger("scriptLine"_s, scriptLine);
    data->setInteger("scriptColumn"_s, scriptColumn);
    return data;
}

Ref<JSON::Object> TimelineRecordFactory::createConsoleProfileData(const String& title)
{
    Ref<JSON::Object> data = JSON::Object::create();
    data->setString("title"_s, title);
    return data;
}

Ref<JSON::Object> TimelineRecordFactory::createProbeSampleData(JSC::BreakpointActionID actionID, unsigned sampleId)
{
    Ref<JSON::Object> data = JSON::Object::create();
    data->setInteger("probeId"_s, actionID);
    data->setInteger("sampleId"_s, sampleId);
    return data;
}

Ref<JSON::Object> TimelineRecordFactory::createEventDispatchData(const Event& event)
{
    Ref<JSON::Object> data = JSON::Object::create();
    data->setString("type"_s, event.type().string());
    return data;
}

Ref<JSON::Object> TimelineRecordFactory::createGenericTimerData(int timerId)
{
    Ref<JSON::Object> data = JSON::Object::create();
    data->setInteger("timerId"_s, timerId);
    return data;
}

Ref<JSON::Object> TimelineRecordFactory::createTimerInstallData(int timerId, Seconds timeout, bool singleShot)
{
    Ref<JSON::Object> data = JSON::Object::create();
    data->setInteger("timerId"_s, timerId);
    data->setInteger("timeout"_s, (int)timeout.milliseconds());
    data->setBoolean("singleShot"_s, singleShot);
    return data;
}

Ref<JSON::Object> TimelineRecordFactory::createEvaluateScriptData(const String& url, int lineNumber, int columnNumber)
{
    Ref<JSON::Object> data = JSON::Object::create();
    data->setString("url"_s, url);
    data->setInteger("lineNumber"_s, lineNumber);
    data->setInteger("columnNumber"_s, columnNumber);
    return data;
}

Ref<JSON::Object> TimelineRecordFactory::createTimeStampData(const String& message)
{
    Ref<JSON::Object> data = JSON::Object::create();
    data->setString("message"_s, message);
    return data;
}

Ref<JSON::Object> TimelineRecordFactory::createAnimationFrameData(int callbackId)
{
    Ref<JSON::Object> data = JSON::Object::create();
    data->setInteger("id"_s, callbackId);
    return data;
}

Ref<JSON::Object> TimelineRecordFactory::createObserverCallbackData(const String& callbackType)
{
    Ref<JSON::Object> data = JSON::Object::create();
    data->setString("type"_s, callbackType);
    return data;
}

static Ref<JSON::Array> createQuad(const FloatQuad& quad)
{
    Ref<JSON::Array> array = JSON::Array::create();
    array->pushDouble(quad.p1().x());
    array->pushDouble(quad.p1().y());
    array->pushDouble(quad.p2().x());
    array->pushDouble(quad.p2().y());
    array->pushDouble(quad.p3().x());
    array->pushDouble(quad.p3().y());
    array->pushDouble(quad.p4().x());
    array->pushDouble(quad.p4().y());
    return array;
}

Ref<JSON::Object> TimelineRecordFactory::createPaintData(const FloatQuad& quad)
{
    Ref<JSON::Object> data = JSON::Object::create();
    data->setArray("clip"_s, createQuad(quad));
    return data;
}

Ref<JSON::Object> TimelineRecordFactory::createScreenshotData(const String& imageData)
{
    Ref<JSON::Object> data = JSON::Object::create();
    data->setString("imageData"_s, imageData);
    return data;
}

void TimelineRecordFactory::appendLayoutRoot(JSON::Object& data, const FloatQuad& quad)
{
    data.setArray("root"_s, createQuad(quad));
}

} // namespace WebCore
