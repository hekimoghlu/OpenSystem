/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 30, 2022.
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
#include "ManagedMediaSource.h"

#if ENABLE(MEDIA_SOURCE)

#include "Event.h"
#include "EventNames.h"
#include "MediaSourcePrivate.h"
#include "SourceBufferList.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(ManagedMediaSource);

Ref<ManagedMediaSource> ManagedMediaSource::create(ScriptExecutionContext& context, MediaSourceInit&& options)
{
    auto mediaSource = adoptRef(*new ManagedMediaSource(context, WTFMove(options)));
    mediaSource->suspendIfNeeded();
    return mediaSource;
}

ManagedMediaSource::ManagedMediaSource(ScriptExecutionContext& context, MediaSourceInit&& options)
    : MediaSource(context, WTFMove(options))
    , m_streamingTimer(*this, &ManagedMediaSource::streamingTimerFired)
{
}

ManagedMediaSource::~ManagedMediaSource()
{
    m_streamingTimer.stop();
}

ExceptionOr<ManagedMediaSource::PreferredQuality> ManagedMediaSource::quality() const
{
    return PreferredQuality::High;
}

bool ManagedMediaSource::isTypeSupported(ScriptExecutionContext& context, const String& type)
{
    return MediaSource::isTypeSupported(context, type);
}

void ManagedMediaSource::elementDetached()
{
    setStreaming(false);
}

void ManagedMediaSource::setStreaming(bool streaming)
{
    if (m_streaming == streaming)
        return;
    ALWAYS_LOG(LOGIDENTIFIER, streaming);
    m_streaming = streaming;
    if (RefPtr msp = protectedPrivate())
        msp->setStreaming(streaming);
    if (streaming) {
        scheduleEvent(eventNames().startstreamingEvent);
        if (m_streamingAllowed) {
            ensurePrefsRead();
            Seconds delay { *m_highThreshold };
            m_streamingTimer.startOneShot(delay);
        }
    } else {
        if (m_streamingTimer.isActive())
            m_streamingTimer.stop();
        scheduleEvent(eventNames().endstreamingEvent);
    }
    notifyElementUpdateMediaState();
}

void ManagedMediaSource::ensurePrefsRead()
{
    ASSERT(scriptExecutionContext());

    if (m_lowThreshold && m_highThreshold)
        return;
    m_lowThreshold = scriptExecutionContext()->settingsValues().managedMediaSourceLowThreshold;
    m_highThreshold = scriptExecutionContext()->settingsValues().managedMediaSourceHighThreshold;
}

void ManagedMediaSource::monitorSourceBuffers()
{
    MediaSource::monitorSourceBuffers();

    if (!activeSourceBuffers()->length()) {
        setStreaming(true);
        return;
    }

    ensurePrefsRead();

    auto currentTime = this->currentTime();
    ASSERT(currentTime.isValid());

    auto limitAhead = [&] (double upper) {
        MediaTime aheadTime = currentTime + MediaTime::createWithDouble(upper);
        return isEnded() ? std::min(duration(), aheadTime) : aheadTime;
    };
    if (!m_streaming) {
        PlatformTimeRanges neededBufferedRange { currentTime, std::max(currentTime, limitAhead(*m_lowThreshold)) };
        if (!isBuffered(neededBufferedRange))
            setStreaming(true);
        return;
    }

    if (auto ahead = limitAhead(*m_highThreshold); currentTime < ahead) {
        if (isBuffered({ currentTime,  ahead }))
            setStreaming(false);
    } else
        setStreaming(false);
}

void ManagedMediaSource::streamingTimerFired()
{
    ALWAYS_LOG(LOGIDENTIFIER, "Disabling streaming due to policy ", *m_highThreshold);
    m_streamingAllowed = false;
    if (RefPtr msp = protectedPrivate())
        msp->setStreamingAllowed(false);
    notifyElementUpdateMediaState();
}

} // namespace WebCore

#endif // ENABLE(MEDIA_SOURCE)
