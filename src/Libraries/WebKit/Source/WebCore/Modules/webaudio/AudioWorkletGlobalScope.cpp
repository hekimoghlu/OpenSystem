/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 30, 2022.
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

#if ENABLE(WEB_AUDIO)
#include "AudioWorkletGlobalScope.h"

#include "AudioParamDescriptor.h"
#include "AudioWorklet.h"
#include "AudioWorkletMessagingProxy.h"
#include "AudioWorkletProcessorConstructionData.h"
#include "BaseAudioContext.h"
#include "CommonVM.h"
#include "JSAudioWorkletProcessor.h"
#include "JSAudioWorkletProcessorConstructor.h"
#include "JSDOMConvert.h"
#include "WebCoreOpaqueRootInlines.h"
#include <JavaScriptCore/JSLock.h>
#include <wtf/CrossThreadCopier.h>
#include <wtf/TZoneMallocInlines.h>
#include <wtf/text/MakeString.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(AudioWorkletGlobalScope);

RefPtr<AudioWorkletGlobalScope> AudioWorkletGlobalScope::tryCreate(AudioWorkletThread& thread, const WorkletParameters& parameters)
{
    auto vm = JSC::VM::tryCreate();
    if (!vm)
        return nullptr;
    auto scope = adoptRef(*new AudioWorkletGlobalScope(thread, vm.releaseNonNull(), parameters));
    scope->addToContextsMap();
    return scope;
}

AudioWorkletGlobalScope::AudioWorkletGlobalScope(AudioWorkletThread& thread, Ref<JSC::VM>&& vm, const WorkletParameters& parameters)
    : WorkletGlobalScope(thread, WTFMove(vm), parameters)
    , m_sampleRate(parameters.sampleRate)
{
    ASSERT(!isMainThread());
}

AudioWorkletGlobalScope::~AudioWorkletGlobalScope() = default;

// https://www.w3.org/TR/webaudio/#dom-audioworkletglobalscope-registerprocessor
ExceptionOr<void> AudioWorkletGlobalScope::registerProcessor(String&& name, Ref<JSAudioWorkletProcessorConstructor>&& processorConstructor)
{
    ASSERT(!isMainThread());

    if (name.isEmpty())
        return Exception { ExceptionCode::NotSupportedError, "Name cannot be the empty string"_s };

    if (m_processorConstructorMap.contains(name))
        return Exception { ExceptionCode::NotSupportedError, "A processor was already registered with this name"_s };

    JSC::JSObject* jsConstructor = processorConstructor->callbackData()->callback();
    auto* globalObject = jsConstructor->globalObject();
    auto& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);

    if (!jsConstructor->isConstructor())
        return Exception { ExceptionCode::TypeError, "Class definition passed to registerProcessor() is not a constructor"_s };

    auto prototype = jsConstructor->getPrototype(vm, globalObject);
    RETURN_IF_EXCEPTION(scope, Exception { ExceptionCode::ExistingExceptionError });

    if (!prototype.isObject())
        return Exception { ExceptionCode::TypeError, "Class definition passed to registerProcessor() has invalid prototype"_s };

    auto parameterDescriptorsValue = jsConstructor->get(globalObject, JSC::Identifier::fromString(vm, "parameterDescriptors"_s));
    RETURN_IF_EXCEPTION(scope, Exception { ExceptionCode::ExistingExceptionError });

    Vector<AudioParamDescriptor> parameterDescriptors;
    if (!parameterDescriptorsValue.isUndefined()) {
        auto parameterDescriptorsConversionResult = convert<IDLSequence<IDLDictionary<AudioParamDescriptor>>>(*globalObject, parameterDescriptorsValue);
        if (UNLIKELY(parameterDescriptorsConversionResult.hasException(scope)))
            return Exception { ExceptionCode::ExistingExceptionError };

        parameterDescriptors = parameterDescriptorsConversionResult.releaseReturnValue();

        HashSet<String> paramNames;
        for (auto& descriptor : parameterDescriptors) {
            auto addResult = paramNames.add(descriptor.name);
            if (!addResult.isNewEntry)
                return Exception { ExceptionCode::NotSupportedError, makeString("parameterDescriptors contain duplicate AudioParam name: "_s, name) };
            if (descriptor.defaultValue < descriptor.minValue)
                return Exception { ExceptionCode::InvalidStateError, makeString("AudioParamDescriptor with name '"_s, name, "' has a defaultValue that is less than the minValue"_s) };
            if (descriptor.defaultValue > descriptor.maxValue)
                return Exception { ExceptionCode::InvalidStateError, makeString("AudioParamDescriptor with name '"_s, name, "' has a defaultValue that is greater than the maxValue"_s) };
        }
    }

    auto addResult = m_processorConstructorMap.add(name, WTFMove(processorConstructor));

    // We've already checked at the beginning of this function but then we ran some JS so we need to check again.
    if (!addResult.isNewEntry)
        return Exception { ExceptionCode::NotSupportedError, "A processor was already registered with this name"_s };

    auto* messagingProxy = thread().messagingProxy();
    if (!messagingProxy)
        return Exception { ExceptionCode::InvalidStateError };

    messagingProxy->postTaskToAudioWorklet([name = WTFMove(name).isolatedCopy(), parameterDescriptors = crossThreadCopy(WTFMove(parameterDescriptors))](AudioWorklet& worklet) mutable {
        ASSERT(isMainThread());
        if (RefPtr audioContext = worklet.audioContext())
            audioContext->addAudioParamDescriptors(name, WTFMove(parameterDescriptors));
    });

    return { };
}

RefPtr<AudioWorkletProcessor> AudioWorkletGlobalScope::createProcessor(const String& name, TransferredMessagePort port, Ref<SerializedScriptValue>&& options)
{
    auto constructor = m_processorConstructorMap.get(name);
    ASSERT(constructor);
    if (!constructor)
        return nullptr;

    JSC::JSObject* jsConstructor = constructor->callbackData()->callback();
    ASSERT(jsConstructor);
    if (!jsConstructor)
        return nullptr;

    auto* globalObject = constructor->callbackData()->globalObject();
    JSC::VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);
    JSC::JSLockHolder lock { globalObject };

    m_pendingProcessorConstructionData = makeUnique<AudioWorkletProcessorConstructionData>(String { name }, MessagePort::entangle(*this, WTFMove(port)));

    JSC::MarkedArgumentBuffer args;
    bool didFail = false;
    auto arg = options->deserialize(*globalObject, globalObject, SerializationErrorMode::NonThrowing, &didFail);
    RETURN_IF_EXCEPTION(scope, nullptr);
    if (didFail)
        return nullptr;
    args.append(arg);
    ASSERT(!args.hasOverflowed());

    auto* object = JSC::construct(globalObject, jsConstructor, args, "Failed to construct AudioWorkletProcessor"_s);
    ASSERT(!!scope.exception() == !object);
    RETURN_IF_EXCEPTION(scope, nullptr);

    auto* jsProcessor = JSC::jsDynamicCast<JSAudioWorkletProcessor*>(object);
    if (!jsProcessor)
        return nullptr;

    m_processors.add(jsProcessor->wrapped());
    return &jsProcessor->wrapped();
}

void AudioWorkletGlobalScope::prepareForDestruction()
{
    m_processorConstructorMap.clear();

    WorkletGlobalScope::prepareForDestruction();
}

std::unique_ptr<AudioWorkletProcessorConstructionData> AudioWorkletGlobalScope::takePendingProcessorConstructionData()
{
    return std::exchange(m_pendingProcessorConstructionData, nullptr);
}

AudioWorkletThread& AudioWorkletGlobalScope::thread() const
{
    return *static_cast<AudioWorkletThread*>(workerOrWorkletThread());
}

void AudioWorkletGlobalScope::handlePreRenderTasks()
{
    // This makes sure that we only drain the MicroTask queue after each render quantum.
    // It is only safe to grab the lock if we are on the context thread.
    RELEASE_ASSERT(isContextThread());
    m_delayMicrotaskDrainingDuringRendering = script()->vm().drainMicrotaskDelayScope();
}

void AudioWorkletGlobalScope::handlePostRenderTasks(size_t currentFrame)
{
    m_currentFrame = currentFrame;

    {
        // Heap allocations are forbidden on the audio thread for performance reasons so we need to
        // explicitly allow the following allocation(s).
        DisableMallocRestrictionsForCurrentThreadScope disableMallocRestrictions;
        // This takes care of processing the MicroTask queue after rendering.
        m_delayMicrotaskDrainingDuringRendering = std::nullopt;
    }
}

void AudioWorkletGlobalScope::processorIsNoLongerNeeded(AudioWorkletProcessor& processor)
{
    m_processors.remove(processor);
}

void AudioWorkletGlobalScope::visitProcessors(JSC::AbstractSlotVisitor& visitor)
{
    m_processors.forEach([&](auto& processor) {
        addWebCoreOpaqueRoot(visitor, processor);
    });
}

} // namespace WebCore

#endif // ENABLE(WEB_AUDIO)
