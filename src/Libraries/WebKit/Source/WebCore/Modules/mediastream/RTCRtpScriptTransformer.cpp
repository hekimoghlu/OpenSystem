/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 27, 2025.
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
#include "RTCRtpScriptTransformer.h"

#if ENABLE(WEB_RTC)

#include "DedicatedWorkerGlobalScope.h"
#include "EventLoop.h"
#include "FrameRateMonitor.h"
#include "JSDOMPromiseDeferred.h"
#include "JSRTCEncodedAudioFrame.h"
#include "JSRTCEncodedVideoFrame.h"
#include "Logging.h"
#include "MessageWithMessagePorts.h"
#include "RTCRtpTransformableFrame.h"
#include "ReadableStream.h"
#include "ReadableStreamSource.h"
#include "SerializedScriptValue.h"
#include "WorkerThread.h"
#include "WritableStream.h"
#include "WritableStreamSink.h"

namespace WebCore {

ExceptionOr<Ref<RTCRtpScriptTransformer>> RTCRtpScriptTransformer::create(ScriptExecutionContext& context, MessageWithMessagePorts&& options)
{
    if (!context.globalObject())
        return Exception { ExceptionCode::InvalidStateError };

    auto& globalObject = *JSC::jsCast<JSDOMGlobalObject*>(context.globalObject());
    JSC::JSLockHolder lock(globalObject.vm());
    auto readableSource = SimpleReadableStreamSource::create();
    auto readable = ReadableStream::create(globalObject, readableSource.copyRef());
    if (readable.hasException())
        return readable.releaseException();
    if (!options.message)
        return Exception { ExceptionCode::InvalidStateError };

    auto ports = MessagePort::entanglePorts(context, WTFMove(options.transferredPorts));
    auto transformer = adoptRef(*new RTCRtpScriptTransformer(context, options.message.releaseNonNull(), WTFMove(ports), readable.releaseReturnValue(), WTFMove(readableSource)));
    transformer->suspendIfNeeded();
    return transformer;
}

RTCRtpScriptTransformer::RTCRtpScriptTransformer(ScriptExecutionContext& context, Ref<SerializedScriptValue>&& options, Vector<Ref<MessagePort>>&& ports, Ref<ReadableStream>&& readable, Ref<SimpleReadableStreamSource>&& readableSource)
    : ActiveDOMObject(&context)
    , m_options(WTFMove(options))
    , m_ports(WTFMove(ports))
    , m_readableSource(WTFMove(readableSource))
    , m_readable(WTFMove(readable))
#if !RELEASE_LOG_DISABLED
    , m_enableAdditionalLogging(context.settingsValues().webRTCMediaPipelineAdditionalLoggingEnabled)
    , m_identifier(RTCRtpScriptTransformerIdentifier::generate())
#endif
{
}

RTCRtpScriptTransformer::~RTCRtpScriptTransformer() = default;

ReadableStream& RTCRtpScriptTransformer::readable()
{
    return m_readable.get();
}

ExceptionOr<Ref<WritableStream>> RTCRtpScriptTransformer::writable()
{
    if (!m_writable) {
        RefPtr context = downcast<WorkerGlobalScope>(scriptExecutionContext());
        if (!context || !context->globalObject())
            return Exception { ExceptionCode::InvalidStateError };

        auto& globalObject = *JSC::jsCast<JSDOMGlobalObject*>(context->globalObject());
        auto writableOrException = WritableStream::create(globalObject, SimpleWritableStreamSink::create([transformer = Ref { *this }](auto& context, auto value) -> ExceptionOr<void> {
            if (!transformer->m_backend)
                return Exception { ExceptionCode::InvalidStateError };

            auto& globalObject = *context.globalObject();
            auto scope = DECLARE_THROW_SCOPE(globalObject.vm());

            auto frameConversionResult = convert<IDLUnion<IDLInterface<RTCEncodedAudioFrame>, IDLInterface<RTCEncodedVideoFrame>>>(globalObject, value);
            if (UNLIKELY(frameConversionResult.hasException(scope)))
                return Exception { ExceptionCode::ExistingExceptionError };

            auto frame = frameConversionResult.releaseReturnValue();
            auto rtcFrame = WTF::switchOn(frame, [&](RefPtr<RTCEncodedAudioFrame>& value) {
                return value->rtcFrame();
            }, [&](RefPtr<RTCEncodedVideoFrame>& value) {
                return value->rtcFrame();
            });

            // If no data, skip the frame since there is nothing to packetize or decode.
            if (rtcFrame->data().data()) {
#if !RELEASE_LOG_DISABLED
                if (transformer->m_enableAdditionalLogging && transformer->m_backend->mediaType() == RTCRtpTransformBackend::MediaType::Video) {
                    if (!transformer->m_writableFrameRateMonitor) {
                        transformer->m_writableFrameRateMonitor = makeUnique<FrameRateMonitor>([identifier = transformer->m_identifier](auto info) {
                            RELEASE_LOG(WebRTC, "RTCRtpScriptTransformer writable %" PRIu64 ", frame at %f, previous frame was at %f, observed frame rate is %f, delay since last frame is %f ms, frame count is %lu", identifier.toUInt64(), info.frameTime.secondsSinceEpoch().value(), info.lastFrameTime.secondsSinceEpoch().value(), info.observedFrameRate, ((info.frameTime - info.lastFrameTime) * 1000).value(), info.frameCount);
                        });
                    }
                    transformer->m_writableFrameRateMonitor->update();
                }
#endif
                transformer->m_backend->processTransformedFrame(rtcFrame.get());
            }
            return { };
        }));
        if (writableOrException.hasException())
            return writableOrException;
        m_writable = writableOrException.releaseReturnValue();
    }
    return Ref { *m_writable };
}

void RTCRtpScriptTransformer::start(Ref<RTCRtpTransformBackend>&& backend)
{
    m_backend = WTFMove(backend);

    auto& context = downcast<WorkerGlobalScope>(*scriptExecutionContext());
    m_backend->setTransformableFrameCallback([weakThis = WeakPtr { *this }, thread = Ref { context.thread() }](Ref<RTCRtpTransformableFrame>&& frame) mutable {
        thread->runLoop().postTaskForMode([weakThis, frame = WTFMove(frame)](auto& context) mutable {
            if (weakThis)
                weakThis->enqueueFrame(context, WTFMove(frame));
        }, WorkerRunLoop::defaultMode());
    });
}

void RTCRtpScriptTransformer::clear(ClearCallback clearCallback)
{
    if (m_backend && clearCallback == ClearCallback::Yes)
        m_backend->clearTransformableFrameCallback();
    m_backend = nullptr;
    stopPendingActivity();
}

void RTCRtpScriptTransformer::enqueueFrame(ScriptExecutionContext& context, Ref<RTCRtpTransformableFrame>&& frame)
{
    if (!m_backend)
        return;

    auto* globalObject = JSC::jsCast<JSDOMGlobalObject*>(context.globalObject());
    if (!globalObject)
        return;

    auto& vm = globalObject->vm();
    JSC::JSLockHolder lock(vm);

    bool isVideo = m_backend->mediaType() == RTCRtpTransformBackend::MediaType::Video;
    if (isVideo && !m_pendingKeyFramePromises.isEmpty() && frame->isKeyFrame()) {
        for (auto& promise : std::exchange(m_pendingKeyFramePromises, { }))
            promise->resolve();
    }

#if !RELEASE_LOG_DISABLED
    if (m_enableAdditionalLogging && isVideo) {
        if (!m_readableFrameRateMonitor) {
            m_readableFrameRateMonitor = makeUnique<FrameRateMonitor>([identifier = m_identifier](auto info) {
                RELEASE_LOG(WebRTC, "RTCRtpScriptTransformer readable %" PRIu64 ", frame at %f, previous frame was at %f, observed frame rate is %f, delay since last frame is %f ms, frame count is %lu", identifier.toUInt64(), info.frameTime.secondsSinceEpoch().value(), info.lastFrameTime.secondsSinceEpoch().value(), info.observedFrameRate, ((info.frameTime - info.lastFrameTime) * 1000).value(), info.frameCount);
            });
        }
        m_readableFrameRateMonitor->update();
    }
#endif

    auto value = isVideo ? toJS(globalObject, globalObject, RTCEncodedVideoFrame::create(WTFMove(frame))) : toJS(globalObject, globalObject, RTCEncodedAudioFrame::create(WTFMove(frame)));
    m_readableSource->enqueue(value);
}

void RTCRtpScriptTransformer::generateKeyFrame(Ref<DeferredPromise>&& promise)
{
    RefPtr context = scriptExecutionContext();
    if (!context || !m_backend || m_backend->side() != RTCRtpTransformBackend::Side::Sender || m_backend->mediaType() != RTCRtpTransformBackend::MediaType::Video) {
        promise->reject(Exception { ExceptionCode::InvalidStateError, "Not attached to a valid video sender"_s });
        return;
    }

    bool shouldRequestKeyFrame = m_pendingKeyFramePromises.isEmpty();
    m_pendingKeyFramePromises.append(WTFMove(promise));
    if (shouldRequestKeyFrame)
        m_backend->requestKeyFrame();
}

void RTCRtpScriptTransformer::sendKeyFrameRequest(Ref<DeferredPromise>&& promise)
{
    RefPtr context = scriptExecutionContext();
    if (!context || !m_backend || m_backend->side() != RTCRtpTransformBackend::Side::Receiver || m_backend->mediaType() != RTCRtpTransformBackend::MediaType::Video) {
        promise->reject(Exception { ExceptionCode::InvalidStateError, "Not attached to a valid video receiver"_s });
        return;
    }

    // FIXME: We should be able to know when the FIR request is sent to resolve the promise at this exact time.
    m_backend->requestKeyFrame();

    context->eventLoop().queueTask(TaskSource::Networking, [promise = WTFMove(promise)]() mutable {
        promise->resolve();
    });
}

JSC::JSValue RTCRtpScriptTransformer::options(JSC::JSGlobalObject& globalObject)
{
    return m_options->deserialize(globalObject, &globalObject, m_ports);
}

} // namespace WebCore

#endif // ENABLE(WEB_RTC)
