/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 21, 2023.
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
#include "WebCodecsVideoDecoder.h"

#if ENABLE(WEB_CODECS)

#include "CSSStyleImageValue.h"
#include "ContextDestructionObserverInlines.h"
#include "DOMException.h"
#include "HTMLCanvasElement.h"
#include "HTMLImageElement.h"
#include "HTMLVideoElement.h"
#include "ImageBitmap.h"
#include "JSDOMPromiseDeferred.h"
#include "JSWebCodecsVideoDecoderSupport.h"
#include "OffscreenCanvas.h"
#include "SVGImageElement.h"
#include "ScriptExecutionContext.h"
#include "WebCodecsControlMessage.h"
#include "WebCodecsEncodedVideoChunk.h"
#include "WebCodecsErrorCallback.h"
#include "WebCodecsUtilities.h"
#include "WebCodecsVideoFrame.h"
#include "WebCodecsVideoFrameOutputCallback.h"
#include <variant>
#include <wtf/ASCIICType.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(WebCodecsVideoDecoder);

Ref<WebCodecsVideoDecoder> WebCodecsVideoDecoder::create(ScriptExecutionContext& context, Init&& init)
{
    auto decoder = adoptRef(*new WebCodecsVideoDecoder(context, WTFMove(init)));
    decoder->suspendIfNeeded();
    return decoder;
}

WebCodecsVideoDecoder::WebCodecsVideoDecoder(ScriptExecutionContext& context, Init&& init)
    : WebCodecsBase(context)
    , m_output(init.output.releaseNonNull())
    , m_error(init.error.releaseNonNull())
{
}

WebCodecsVideoDecoder::~WebCodecsVideoDecoder() = default;

static bool isSupportedDecoderCodec(const String& codec, const Settings::Values& settings)
{
    return codec.startsWith("vp8"_s) || codec.startsWith("vp09.00"_s) || codec.startsWith("avc1."_s)
#if ENABLE(WEB_RTC)
        || (codec.startsWith("vp09.02"_s) && settings.webRTCVP9Profile2CodecEnabled)
#endif
        || (codec.startsWith("hev1."_s) && settings.webCodecsHEVCEnabled)
        || (codec.startsWith("hvc1."_s) && settings.webCodecsHEVCEnabled)
        || (codec.startsWith("av01.0"_s) && settings.webCodecsAV1Enabled);
}

static bool isValidDecoderConfig(const WebCodecsVideoDecoderConfig& config)
{
    // https://w3c.github.io/webcodecs/#valid-videodecoderconfig
    // 1. If codec is empty after stripping leading and trailing ASCII whitespace, return false.
    if (StringView(config.codec).trim(isASCIIWhitespace<UChar>).isEmpty())
        return false;

    // 2. If one of codedWidth or codedHeight is provided but the other isnâ€™t, return false.
    if (!!config.codedWidth != !!config.codedHeight)
        return false;

    // 3. If codedWidth = 0 or codedHeight = 0, return false.
    if (config.codedWidth && !*config.codedWidth)
        return false;
    if (config.codedHeight && !*config.codedHeight)
        return false;

    // 4. If one of displayAspectWidth or displayAspectHeight is provided but the other isnâ€™t, return false.
    if (!!config.displayAspectWidth != !!config.displayAspectHeight)
        return false;

    // 5. If displayAspectWidth = 0 or displayAspectHeight = 0, return false.
    if (config.displayAspectWidth && !*config.displayAspectWidth)
        return false;
    if (config.displayAspectHeight && !*config.displayAspectHeight)
        return false;

    // 6. If description is [detached], return false.
    if (config.description && std::visit([](auto& view) { return view->isDetached(); }, *config.description))
        return false;

    // 7. Return true.
    return true;
}

static VideoDecoder::Config createVideoDecoderConfig(const WebCodecsVideoDecoderConfig& config)
{
    Vector<uint8_t> description;
    if (config.description) {
        auto data = std::visit([](auto& buffer) {
            return buffer ? buffer->span() : std::span<const uint8_t> { };
        }, *config.description);
        if (!data.empty())
            description = data;
    }

    return {
        .description = description,
        .width = config.codedWidth.value_or(0),
        .height = config.codedHeight.value_or(0),
        .colorSpace = config.colorSpace,
        .decoding = config.hardwareAcceleration == HardwareAcceleration::PreferSoftware ? VideoDecoder::HardwareAcceleration::No : VideoDecoder::HardwareAcceleration::Yes
    };
}

ExceptionOr<void> WebCodecsVideoDecoder::configure(ScriptExecutionContext& context, WebCodecsVideoDecoderConfig&& config)
{
    if (!isValidDecoderConfig(config))
        return Exception { ExceptionCode::TypeError, "Config is not valid"_s };

    if (state() == WebCodecsCodecState::Closed || !scriptExecutionContext())
        return Exception { ExceptionCode::InvalidStateError, "VideoDecoder is closed"_s };

    setState(WebCodecsCodecState::Configured);
    m_isKeyChunkRequired = true;

    bool isSupportedCodec = isSupportedDecoderCodec(config.codec, context.settingsValues());
    queueControlMessageAndProcess({ *this, [this, codec = config.codec, config = createVideoDecoderConfig(config), isSupportedCodec]() mutable {
        RefPtr context = scriptExecutionContext();

        auto identifier = context->identifier();

        blockControlMessageQueue();
        if (!isSupportedCodec) {
            postTaskToCodec<WebCodecsVideoDecoder>(identifier, *this, [] (auto& decoder) {
                decoder.closeDecoder(Exception { ExceptionCode::NotSupportedError, "Codec is not supported"_s });
            });
            return WebCodecsControlMessageOutcome::Processed;
        }

        Ref createDecoderPromise = VideoDecoder::create(codec, WTFMove(config), [identifier, weakThis = ThreadSafeWeakPtr { *this }, decoderCount = ++m_decoderCount] (auto&& result) {
            postTaskToCodec<WebCodecsVideoDecoder>(identifier, weakThis, [result = WTFMove(result), decoderCount] (auto& decoder) mutable {
                if (decoder.state() != WebCodecsCodecState::Configured || decoder.m_decoderCount != decoderCount)
                    return;

                if (!result) {
                    decoder.closeDecoder(Exception { ExceptionCode::EncodingError, WTFMove(result).error() });
                    return;
                }

                auto decodedResult = WTFMove(result).value();
                WebCodecsVideoFrame::BufferInit init;
                init.codedWidth = decodedResult.frame->presentationSize().width();
                init.codedHeight = decodedResult.frame->presentationSize().height();
                init.timestamp = decodedResult.timestamp;
                init.duration = decodedResult.duration;
                init.colorSpace = decodedResult.frame->colorSpace();

                auto videoFrame = WebCodecsVideoFrame::create(*decoder.scriptExecutionContext(), WTFMove(decodedResult.frame), WTFMove(init));
                decoder.m_output->handleEvent(WTFMove(videoFrame));
            });
        });

        context->enqueueTaskWhenSettled(WTFMove(createDecoderPromise), TaskSource::MediaElement, [weakThis = ThreadSafeWeakPtr { *this }] (auto&& result) mutable {
            auto protectedThis = weakThis.get();
            if (!protectedThis)
                return;
            if (!result) {
                protectedThis->closeDecoder(Exception { ExceptionCode::NotSupportedError, WTFMove(result.error()) });
                return;
            }
            protectedThis->setInternalDecoder(WTFMove(*result));
            protectedThis->unblockControlMessageQueue();
        });

        return WebCodecsControlMessageOutcome::Processed;
    } });
    return { };
}

ExceptionOr<void> WebCodecsVideoDecoder::decode(Ref<WebCodecsEncodedVideoChunk>&& chunk)
{
    if (state() != WebCodecsCodecState::Configured)
        return Exception { ExceptionCode::InvalidStateError, "VideoDecoder is not configured"_s };

    if (m_isKeyChunkRequired) {
        if (chunk->type() != WebCodecsEncodedVideoChunkType::Key)
            return Exception { ExceptionCode::DataError, "Key frame is required"_s };
        m_isKeyChunkRequired = false;
    }

    queueCodecControlMessageAndProcess({ *this, [this, chunk = WTFMove(chunk)]() mutable {
        incrementCodecOperationCount();
        Ref internalDecoder = *m_internalDecoder;
        protectedScriptExecutionContext()->enqueueTaskWhenSettled(internalDecoder->decode({ chunk->span(), chunk->type() == WebCodecsEncodedVideoChunkType::Key, chunk->timestamp(), chunk->duration() }), TaskSource::MediaElement, [weakThis = ThreadSafeWeakPtr { * this }, pendingActivity = makePendingActivity(*this)] (auto&& result) {
            RefPtr protectedThis = weakThis.get();
            if (!protectedThis)
                return;

            if (!result) {
                protectedThis->closeDecoder(Exception { ExceptionCode::EncodingError, WTFMove(result.error()) });
                return;
            }
            protectedThis->decrementCodecOperationCountAndMaybeProcessControlMessageQueue();
        });
        return WebCodecsControlMessageOutcome::Processed;
    } });
    return { };
}

ExceptionOr<void> WebCodecsVideoDecoder::flush(Ref<DeferredPromise>&& promise)
{
    if (state() != WebCodecsCodecState::Configured)
        return Exception { ExceptionCode::InvalidStateError, "VideoDecoder is not configured"_s };

    m_isKeyChunkRequired = true;
    m_pendingFlushPromises.append(promise);
    queueControlMessageAndProcess({ *this, [this, promise = WTFMove(promise)]() mutable {
        Ref internalDecoder = *m_internalDecoder;
        protectedScriptExecutionContext()->enqueueTaskWhenSettled(internalDecoder->flush(), TaskSource::MediaElement, [weakThis = ThreadSafeWeakPtr { *this }, pendingActivity = makePendingActivity(*this), promise = WTFMove(promise)] (auto&&) {
            promise->resolve();
            if (RefPtr protectedThis = weakThis.get())
                protectedThis->m_pendingFlushPromises.removeFirstMatching([&](auto& flushPromise) { return promise.ptr() == flushPromise.ptr(); });
        });
        return WebCodecsControlMessageOutcome::Processed;
    } });
    return { };
}

ExceptionOr<void> WebCodecsVideoDecoder::reset()
{
    return resetDecoder(Exception { ExceptionCode::AbortError, "Reset called"_s });
}

ExceptionOr<void> WebCodecsVideoDecoder::close()
{
    return closeDecoder(Exception { ExceptionCode::AbortError, "Close called"_s });
}

void WebCodecsVideoDecoder::isConfigSupported(ScriptExecutionContext& context, WebCodecsVideoDecoderConfig&& config, Ref<DeferredPromise>&& promise)
{
    if (!isValidDecoderConfig(config)) {
        promise->reject(Exception { ExceptionCode::TypeError, "Config is not valid"_s });
        return;
    }

    if (!isSupportedDecoderCodec(config.codec, context.settingsValues())) {
        promise->template resolve<IDLDictionary<WebCodecsVideoDecoderSupport>>(WebCodecsVideoDecoderSupport { false, WTFMove(config) });
        return;
    }

    Ref createDecoderPromise = VideoDecoder::create(config.codec, createVideoDecoderConfig(config), [] (auto&&) { });
    context.enqueueTaskWhenSettled(WTFMove(createDecoderPromise), TaskSource::MediaElement, [config = WTFMove(config), promise = WTFMove(promise)](auto&& result) mutable {
        promise->template resolve<IDLDictionary<WebCodecsVideoDecoderSupport>>(WebCodecsVideoDecoderSupport { !!result, WTFMove(config) });
    });
}

ExceptionOr<void> WebCodecsVideoDecoder::closeDecoder(Exception&& exception)
{
    auto result = resetDecoder(exception);
    if (result.hasException())
        return result;
    setState(WebCodecsCodecState::Closed);
    m_internalDecoder = nullptr;
    if (exception.code() != ExceptionCode::AbortError)
        m_error->handleEvent(DOMException::create(WTFMove(exception)));

    return { };
}

ExceptionOr<void> WebCodecsVideoDecoder::resetDecoder(const Exception& exception)
{
    if (state() == WebCodecsCodecState::Closed)
        return Exception { ExceptionCode::InvalidStateError, "VideoDecoder is closed"_s };

    setState(WebCodecsCodecState::Unconfigured);
    if (RefPtr internalDecoder = std::exchange(m_internalDecoder, { }))
        internalDecoder->reset();
    clearControlMessageQueueAndMaybeScheduleDequeueEvent();

    auto promises = std::exchange(m_pendingFlushPromises, { });
    for (auto& promise : promises)
        promise->reject(exception);

    return { };
}

void WebCodecsVideoDecoder::setInternalDecoder(Ref<VideoDecoder>&& internalDecoder)
{
    m_internalDecoder = WTFMove(internalDecoder);
}

void WebCore::WebCodecsVideoDecoder::suspend(ReasonForSuspension)
{
}

void WebCodecsVideoDecoder::stop()
{
    setState(WebCodecsCodecState::Closed);
    m_internalDecoder = nullptr;
    clearControlMessageQueue();
    m_pendingFlushPromises.clear();
}

} // namespace WebCore

#endif // ENABLE(WEB_CODECS)
