/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 19, 2024.
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
#include "PerformanceMeasure.h"

#include "SerializedScriptValue.h"

namespace WebCore {

ExceptionOr<Ref<PerformanceMeasure>> PerformanceMeasure::create(const String& name, double startTime, double endTime, Ref<SerializedScriptValue>&& serializedDetail)
{
    return adoptRef(*new PerformanceMeasure(name, startTime, endTime, WTFMove(serializedDetail)));
}

PerformanceMeasure::PerformanceMeasure(const String& name, double startTime, double endTime, Ref<SerializedScriptValue>&& serializedDetail)
    : PerformanceEntry(name, startTime, endTime)
    , m_serializedDetail(WTFMove(serializedDetail))
{
}

PerformanceMeasure::~PerformanceMeasure() = default;

JSC::JSValue PerformanceMeasure::detail(JSC::JSGlobalObject& globalObject)
{
    return m_serializedDetail->deserialize(globalObject, &globalObject);
}

}

