/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 25, 2024.
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
#include "AudioMediaStreamTrackRendererInternalUnit.h"

#if ENABLE(MEDIA_STREAM)

#include "AudioMediaStreamTrackRenderer.h"
#include "AudioSampleDataSource.h"
#include "AudioSession.h"
#include "CAAudioStreamDescription.h"
#include "CoreAudioCaptureDevice.h"
#include "CoreAudioCaptureDeviceManager.h"
#include "Logging.h"
#include "SpanCoreAudio.h"
#include <Accelerate/Accelerate.h>
#include <pal/spi/cocoa/AudioToolboxSPI.h>
#include <wtf/Lock.h>
#include <wtf/RefCounted.h>
#include <wtf/TZoneMallocInlines.h>

#include <pal/cf/AudioToolboxSoftLink.h>
#include <pal/cf/CoreMediaSoftLink.h>

namespace WebCore {

class LocalAudioMediaStreamTrackRendererInternalUnit final : public AudioMediaStreamTrackRendererInternalUnit, public RefCounted<LocalAudioMediaStreamTrackRendererInternalUnit>  {
    WTF_MAKE_TZONE_ALLOCATED_INLINE(LocalAudioMediaStreamTrackRendererInternalUnit);
public:
    static Ref<AudioMediaStreamTrackRendererInternalUnit> create(const String& deviceID, Client& client)
    {
        auto unit = adoptRef(*new LocalAudioMediaStreamTrackRendererInternalUnit(client));
        unit->setAudioOutputDevice(deviceID);
        return unit;
    }

    void ref() const final { RefCounted::ref(); }
    void deref() const final { RefCounted::deref(); }

private:
    explicit LocalAudioMediaStreamTrackRendererInternalUnit(Client&);
    void createAudioUnitIfNeeded();

    // AudioMediaStreamTrackRendererInternalUnit API.
    void start() final;
    void stop() final;
    void retrieveFormatDescription(CompletionHandler<void(std::optional<CAAudioStreamDescription>)>&&) final;
    void setAudioOutputDevice(const String&);

    OSStatus render(AudioUnitRenderActionFlags*, const AudioTimeStamp*, UInt32 sampleCount, AudioBufferList*);
    static OSStatus renderingCallback(void*, AudioUnitRenderActionFlags*, const AudioTimeStamp*, UInt32 inBusNumber, UInt32 sampleCount, AudioBufferList*);

    ThreadSafeWeakPtr<Client> m_client;
    std::optional<CAAudioStreamDescription> m_outputDescription;
    AudioComponentInstance m_remoteIOUnit { nullptr };
    bool m_isStarted { false };
    uint64_t m_sampleTime { 0 };
#if PLATFORM(MAC)
    uint32_t m_deviceID { 0 };
#endif
    String m_audioOutputDeviceID;
};


LocalAudioMediaStreamTrackRendererInternalUnit::LocalAudioMediaStreamTrackRendererInternalUnit(Client& client)
    : m_client(client)
{
}

void LocalAudioMediaStreamTrackRendererInternalUnit::retrieveFormatDescription(CompletionHandler<void(std::optional<CAAudioStreamDescription>)>&& callback)
{
    createAudioUnitIfNeeded();
    callback(m_outputDescription);
}

void LocalAudioMediaStreamTrackRendererInternalUnit::setAudioOutputDevice(const String& deviceID)
{
#if PLATFORM(MAC)
    if (deviceID == AudioMediaStreamTrackRenderer::defaultDeviceID())
        return;

    auto device = CoreAudioCaptureDeviceManager::singleton().coreAudioDeviceWithUID(deviceID);

    if (!device && !deviceID.isEmpty()) {
        RELEASE_LOG(WebRTC, "AudioMediaStreamTrackRendererInternalUnit::setAudioOutputDeviceId - did not find device");
        return;
    }

    m_audioOutputDeviceID = deviceID;

    auto audioUnitDeviceID = device ? device->deviceID() : 0;
    if (m_deviceID == audioUnitDeviceID)
        return;

    bool shouldRestart = m_isStarted;
    if (m_isStarted)
        stop();

    m_deviceID = audioUnitDeviceID;

    if (shouldRestart)
        start();
#else
    UNUSED_PARAM(deviceID);
#endif
}

void LocalAudioMediaStreamTrackRendererInternalUnit::start()
{
    RELEASE_LOG_INFO(WebRTC, "LocalAudioMediaStreamTrackRendererInternalUnit::start");
    if (m_isStarted)
        return;

    createAudioUnitIfNeeded();
    if (!m_remoteIOUnit)
        return;

    m_sampleTime = 0;
    if (auto error = PAL::AudioOutputUnitStart(m_remoteIOUnit)) {
        RELEASE_LOG_ERROR(WebRTC, "AudioMediaStreamTrackRendererInternalUnit::start AudioOutputUnitStart failed, error = %d", error);
        PAL::AudioComponentInstanceDispose(m_remoteIOUnit);
        m_remoteIOUnit = nullptr;
        return;
    }
    m_isStarted = true;
    RELEASE_LOG(WebRTC, "AudioMediaStreamTrackRendererInternalUnit is started");
}

void LocalAudioMediaStreamTrackRendererInternalUnit::stop()
{
    RELEASE_LOG_INFO(WebRTC, "LocalAudioMediaStreamTrackRendererInternalUnit::stop");
    if (!m_remoteIOUnit)
        return;

    if (m_isStarted) {
        PAL::AudioOutputUnitStop(m_remoteIOUnit);
        m_isStarted = false;
    }

    PAL::AudioComponentInstanceDispose(m_remoteIOUnit);
    m_remoteIOUnit = nullptr;
}

void LocalAudioMediaStreamTrackRendererInternalUnit::createAudioUnitIfNeeded()
{
    ASSERT(!m_remoteIOUnit || m_outputDescription);
    if (m_remoteIOUnit)
        return;

    AudioComponentInstance remoteIOUnit { nullptr };

    AudioComponentDescription ioUnitDescription { kAudioUnitType_Output, 0, kAudioUnitManufacturer_Apple, 0, 0 };
#if PLATFORM(IOS_FAMILY)
    ioUnitDescription.componentSubType = kAudioUnitSubType_RemoteIO;
#else
    ioUnitDescription.componentSubType = kAudioUnitSubType_DefaultOutput;
#endif

    AudioComponent ioComponent = PAL::AudioComponentFindNext(nullptr, &ioUnitDescription);
    ASSERT(ioComponent);
    if (!ioComponent) {
        RELEASE_LOG_ERROR(WebRTC, "AudioMediaStreamTrackRendererInternalUnit::createAudioUnit unable to find remote IO unit component");
        return;
    }

    auto error = PAL::AudioComponentInstanceNew(ioComponent, &remoteIOUnit);
    if (error) {
        RELEASE_LOG_ERROR(WebRTC, "AudioMediaStreamTrackRendererInternalUnit::createAudioUnit unable to open vpio unit, error = %d", error);
        return;
    }

#if PLATFORM(IOS_FAMILY)
    UInt32 param = 1;
    error = PAL::AudioUnitSetProperty(remoteIOUnit, kAudioOutputUnitProperty_EnableIO, kAudioUnitScope_Output, 0, &param, sizeof(param));
    if (error) {
        RELEASE_LOG_ERROR(WebRTC, "AudioMediaStreamTrackRendererInternalUnit::createAudioUnit unable to enable vpio unit output, error = %d", error);
        return;
    }
#endif

#if PLATFORM(MAC)
    if (m_deviceID) {
        error = PAL::AudioUnitSetProperty(remoteIOUnit, kAudioOutputUnitProperty_CurrentDevice, kAudioUnitScope_Global, 0, &m_deviceID, sizeof(m_deviceID));
        if (error) {
            RELEASE_LOG_ERROR(WebRTC, "AudioMediaStreamTrackRendererInternalUnit::createAudioUnit unable to set unit device ID %d, error %d (%.4s)", (int)m_deviceID, (int)error, (char*)&error);
            return;
        }
    }
#endif

    AURenderCallbackStruct callback = { LocalAudioMediaStreamTrackRendererInternalUnit::renderingCallback, this };
    error = PAL::AudioUnitSetProperty(remoteIOUnit, kAudioUnitProperty_SetRenderCallback, kAudioUnitScope_Global, 0, &callback, sizeof(callback));
    if (error) {
        RELEASE_LOG_ERROR(WebRTC, "AudioMediaStreamTrackRendererInternalUnit::createAudioUnit unable to set vpio unit speaker proc, error = %d", error);
        return;
    }

    if (!m_outputDescription) {
        AudioStreamBasicDescription outputDescription { };
        UInt32 size = sizeof(outputDescription);
        error  = PAL::AudioUnitGetProperty(remoteIOUnit, kAudioUnitProperty_StreamFormat, kAudioUnitScope_Input, 0, &outputDescription, &size);
        if (error) {
            RELEASE_LOG_ERROR(WebRTC, "AudioMediaStreamTrackRendererInternalUnit::createAudioUnit unable to get input stream format, error = %d", error);
            return;
        }

        outputDescription.mSampleRate = AudioSession::protectedSharedSession()->sampleRate();
        m_outputDescription = outputDescription;
    }
    error = PAL::AudioUnitSetProperty(remoteIOUnit, kAudioUnitProperty_StreamFormat, kAudioUnitScope_Input, 0, &m_outputDescription->streamDescription(), sizeof(m_outputDescription->streamDescription()));
    if (error) {
        RELEASE_LOG_ERROR(WebRTC, "AudioMediaStreamTrackRendererInternalUnit::createAudioUnit unable to set input stream format, error = %d", error);
        return;
    }

    error = PAL::AudioUnitInitialize(remoteIOUnit);
    if (error) {
        RELEASE_LOG_ERROR(WebRTC, "AudioMediaStreamTrackRendererInternalUnit::createAudioUnit AudioUnitInitialize() failed, error = %d", error);
        return;
    }
    m_remoteIOUnit = remoteIOUnit;
}

static void clipAudioBuffer(std::span<float> span)
{
    float minimum = -1;
    float maximum = 1;
    vDSP_vclip(span.data(), 1, &minimum, &maximum, span.data(), 1, span.size());
}

static void clipAudioBuffer(std::span<double> span)
{
    double minimum = -1;
    double maximum = 1;
    vDSP_vclipD(span.data(), 1, &minimum, &maximum, span.data(), 1, span.size());
}

static void clipAudioBufferList(AudioBufferList& list, AudioStreamDescription::PCMFormat format)
{
    switch (format) {
    case AudioStreamDescription::Int16:
        break;
    case AudioStreamDescription::Int32:
        break;
    case AudioStreamDescription::Float32:
        for (auto& buffer : span(list))
            clipAudioBuffer(mutableSpan<float>(buffer));
        break;
    case AudioStreamDescription::Float64:
        for (auto& buffer : span(list))
            clipAudioBuffer(mutableSpan<double>(buffer));
        break;
    case AudioStreamDescription::Uint8:
    case AudioStreamDescription::Int24:
    case AudioStreamDescription::None:
        ASSERT_NOT_REACHED();
        break;
    }
}

OSStatus LocalAudioMediaStreamTrackRendererInternalUnit::render(AudioUnitRenderActionFlags* actionFlags, const AudioTimeStamp* timeStamp, UInt32 sampleCount, AudioBufferList* ioData)
{
    RefPtr client = m_client.get();
    if (!client)
        return kAudio_ParamError;

    auto sampleTime = timeStamp->mSampleTime;
    // If we observe an irregularity in the timeline, we trigger a reset.
    if (m_sampleTime && (m_sampleTime + 2 * sampleCount < sampleTime || sampleTime <= m_sampleTime))
        client->reset();
    m_sampleTime = sampleTime < std::numeric_limits<Float64>::max() - sampleCount ? sampleTime : 0;

    auto result = client->render(sampleCount, *ioData, sampleTime, timeStamp->mHostTime, *actionFlags);
    // FIXME: We should probably introduce a limiter to limit the amount of clipping.
    clipAudioBufferList(*ioData, m_outputDescription->format());
    return result;
}

OSStatus LocalAudioMediaStreamTrackRendererInternalUnit::renderingCallback(void* processor, AudioUnitRenderActionFlags* actionFlags, const AudioTimeStamp* timeStamp, UInt32, UInt32 sampleCount, AudioBufferList* ioData)
{
    return static_cast<LocalAudioMediaStreamTrackRendererInternalUnit*>(processor)->render(actionFlags, timeStamp, sampleCount, ioData);
}

static auto createInternalUnit = LocalAudioMediaStreamTrackRendererInternalUnit::create;

void AudioMediaStreamTrackRendererInternalUnit::setCreateFunction(CreateFunction function)
{
    ASSERT(function);
    createInternalUnit = function;
}

Ref<AudioMediaStreamTrackRendererInternalUnit> AudioMediaStreamTrackRendererInternalUnit::create(const String& deviceID, Client& client)
{
    return createInternalUnit(deviceID, client);
}

} // namespace WebCore

#endif // ENABLE(MEDIA_STREAM)
