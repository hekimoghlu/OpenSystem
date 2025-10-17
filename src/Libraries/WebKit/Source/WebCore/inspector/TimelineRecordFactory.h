/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 22, 2024.
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

#include <JavaScriptCore/DebuggerPrimitives.h>
#include <wtf/Forward.h>
#include <wtf/JSONValues.h>
#include <wtf/Seconds.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

class Event;
class FloatQuad;

class TimelineRecordFactory {
public:
    static Ref<JSON::Object> createGenericRecord(double startTime, int maxCallStackDepth);

    static Ref<JSON::Object> createRenderingFrameData(const String& name);
    static Ref<JSON::Object> createFunctionCallData(const String& scriptName, int scriptLine, int scriptColumn);
    static Ref<JSON::Object> createConsoleProfileData(const String& title);
    static Ref<JSON::Object> createProbeSampleData(JSC::BreakpointActionID, unsigned sampleId);
    static Ref<JSON::Object> createEventDispatchData(const Event&);
    static Ref<JSON::Object> createGenericTimerData(int timerId);
    static Ref<JSON::Object> createTimerInstallData(int timerId, Seconds timeout, bool singleShot);
    static Ref<JSON::Object> createEvaluateScriptData(const String&, int lineNumber, int columnNumber);
    static Ref<JSON::Object> createTimeStampData(const String&);
    static Ref<JSON::Object> createAnimationFrameData(int callbackId);
    static Ref<JSON::Object> createObserverCallbackData(const String& callbackType);
    static Ref<JSON::Object> createPaintData(const FloatQuad&);
    static Ref<JSON::Object> createScreenshotData(const String& imageData);

    static void appendLayoutRoot(JSON::Object& data, const FloatQuad&);

private:
    TimelineRecordFactory() { }
};

} // namespace WebCore
