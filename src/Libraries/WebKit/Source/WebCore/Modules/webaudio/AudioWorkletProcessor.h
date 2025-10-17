/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 7, 2024.
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

#if ENABLE(WEB_AUDIO)
#include "AudioArray.h"
#include "ExceptionOr.h"
#include "JSValueInWrappedObject.h"
#include "ScriptWrappable.h"
#include <wtf/Forward.h>
#include <wtf/Ref.h>
#include <wtf/RobinHoodHashMap.h>
#include <wtf/ThreadSafeWeakPtr.h>
#include <wtf/WeakPtr.h>

namespace JSC {
class JSArray;
template<typename T, size_t, class> class MarkedVector;
using MarkedArgumentBuffer = MarkedVector<JSValue, 8, RecordOverflow>;
}

namespace WebCore {

class AudioBus;
class AudioWorkletGlobalScope;
class AudioWorkletProcessorConstructionData;
class JSCallbackDataStrong;
class MessagePort;
class ScriptExecutionContext;
class WebCoreOpaqueRoot;

class AudioWorkletProcessor : public ScriptWrappable, public ThreadSafeRefCountedAndCanMakeThreadSafeWeakPtr<AudioWorkletProcessor> {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(AudioWorkletProcessor);
public:
    static ExceptionOr<Ref<AudioWorkletProcessor>> create(ScriptExecutionContext&);
    ~AudioWorkletProcessor();

    const String& name() const { return m_name; }
    MessagePort& port() { return m_port.get(); }

    bool process(const Vector<RefPtr<AudioBus>>& inputs, Vector<Ref<AudioBus>>& outputs, const MemoryCompactLookupOnlyRobinHoodHashMap<String, std::unique_ptr<AudioFloatArray>>& paramValuesMap, bool& threwException);

    JSValueInWrappedObject& jsInputsWrapper() { return m_jsInputs; }
    JSValueInWrappedObject& jsOutputsWrapper() { return m_jsOutputs; }
    JSValueInWrappedObject& jsParamValuesWrapper() { return m_jsParamValues; }

private:
    explicit AudioWorkletProcessor(AudioWorkletGlobalScope&, const AudioWorkletProcessorConstructionData&);
    void buildJSArguments(JSC::VM&, JSC::JSGlobalObject&, JSC::MarkedArgumentBuffer&, const Vector<RefPtr<AudioBus>>& inputs, Vector<Ref<AudioBus>>& outputs, const MemoryCompactLookupOnlyRobinHoodHashMap<String, std::unique_ptr<AudioFloatArray>>& paramValuesMap);

    Ref<AudioWorkletGlobalScope> m_globalScope;
    String m_name;
    Ref<MessagePort> m_port;
    JSValueInWrappedObject m_jsInputs;
    JSValueInWrappedObject m_jsOutputs;
    JSValueInWrappedObject m_jsParamValues;
};

WebCoreOpaqueRoot root(AudioWorkletProcessor*);

} // namespace WebCore

#endif // ENABLE(WEB_AUDIO)
