/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 10, 2023.
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

#if ENABLE(MEDIA_SOURCE)

#include "MediaPlayerEnums.h"
#include "SourceBufferPrivateClient.h"
#include <JavaScriptCore/Forward.h>
#include <pal/spi/cocoa/MediaToolboxSPI.h>
#include <variant>
#include <wtf/Expected.h>
#include <wtf/RefCounted.h>
#include <wtf/ThreadSafeRefCounted.h>

namespace WTF {
class Logger;
}

namespace WebCore {

class ContentType;
class MediaSampleAVFObjC;
class SharedBuffer;
struct TrackInfo;

class WEBCORE_EXPORT SourceBufferParser : public ThreadSafeRefCounted<SourceBufferParser, WTF::DestructionThread::Main> {
public:
    static MediaPlayerEnums::SupportsType isContentTypeSupported(const ContentType&);

    static RefPtr<SourceBufferParser> create(const ContentType&);
    virtual ~SourceBufferParser() = default;

    enum class Type : uint8_t {
        AVFObjC,
        WebM,
    };
    virtual Type type() const = 0;
    enum class AppendFlags : uint8_t {
        None,
        Discontinuity,
    };

    class Segment {
    public:
        Segment(Ref<SharedBuffer>&&);
        Segment(Segment&&) = default;
        Ref<SharedBuffer> takeSharedBuffer();
        Ref<SharedBuffer> getData(size_t offset, size_t length) const;

        size_t size() const;

        enum class ReadError { EndOfFile, FatalError };
        using ReadResult = Expected<size_t, ReadError>;

        ReadResult read(std::span<uint8_t> destination, size_t position = 0) const;

    private:
        Ref<SharedBuffer> m_segment;
    };

    using CallOnClientThreadCallback = Function<void(Function<void()>&&)>;
    void setCallOnClientThreadCallback(CallOnClientThreadCallback&&);

    // appendData will be called on the SourceBufferPrivateAVFObjC data parser queue.
    // Other methods will be called on the main thread, but only once appendData has returned.
    virtual Expected<void, PlatformMediaError> appendData(Segment&&, AppendFlags = AppendFlags::None) = 0;
    virtual void flushPendingMediaData() = 0;
    virtual void resetParserState() = 0;
    virtual void invalidate() = 0;
    virtual void setMinimumAudioSampleDuration(float);
#if !RELEASE_LOG_DISABLED
    virtual void setLogger(const Logger&, uint64_t logIdentifier) = 0;
#endif

    // Will be called on the main thread.
    using InitializationSegment = SourceBufferPrivateClient::InitializationSegment;
    using DidParseInitializationDataCallback = Function<void(InitializationSegment&&)>;
    void setDidParseInitializationDataCallback(DidParseInitializationDataCallback&& callback)
    {
        m_didParseInitializationDataCallback = WTFMove(callback);
    }

    // Will be called on the main thread.
    using DidProvideMediaDataCallback = Function<void(Ref<MediaSampleAVFObjC>&&, uint64_t trackID, const String& mediaType)>;
    void setDidProvideMediaDataCallback(DidProvideMediaDataCallback&& callback)
    {
        m_didProvideMediaDataCallback = WTFMove(callback);
    }

    // Will be called synchronously on the parser thead.
    using WillProvideContentKeyRequestInitializationDataForTrackIDCallback = Function<void(uint64_t trackID)>;
    void setWillProvideContentKeyRequestInitializationDataForTrackIDCallback(WillProvideContentKeyRequestInitializationDataForTrackIDCallback&& callback)
    {
        m_willProvideContentKeyRequestInitializationDataForTrackIDCallback = WTFMove(callback);
    }

    // Will be called synchronously on the parser thead.
    using DidProvideContentKeyRequestInitializationDataForTrackIDCallback = Function<void(Ref<SharedBuffer>&&, uint64_t trackID)>;
    void setDidProvideContentKeyRequestInitializationDataForTrackIDCallback(DidProvideContentKeyRequestInitializationDataForTrackIDCallback&& callback)
    {
        m_didProvideContentKeyRequestInitializationDataForTrackIDCallback = WTFMove(callback);
    }

    // Will be called on the main thread.
    using DidProvideContentKeyRequestIdentifierForTrackIDCallback = Function<void(Ref<SharedBuffer>&&, uint64_t trackID)>;
    void setDidProvideContentKeyRequestIdentifierForTrackIDCallback(DidProvideContentKeyRequestIdentifierForTrackIDCallback&& callback)
    {
        m_didProvideContentKeyRequestIdentifierForTrackIDCallback = WTFMove(callback);
    }

    // Will be called on the main thread.
    using DidUpdateFormatDescriptionForTrackIDCallback = Function<void(Ref<TrackInfo>&&, uint64_t trackID)>;
    void setDidUpdateFormatDescriptionForTrackIDCallback(DidUpdateFormatDescriptionForTrackIDCallback&& callback)
    {
        m_didUpdateFormatDescriptionForTrackIDCallback = WTFMove(callback);
    }

protected:
    SourceBufferParser();

    CallOnClientThreadCallback m_callOnClientThreadCallback;
    DidParseInitializationDataCallback m_didParseInitializationDataCallback;
    DidProvideMediaDataCallback m_didProvideMediaDataCallback;
    WillProvideContentKeyRequestInitializationDataForTrackIDCallback m_willProvideContentKeyRequestInitializationDataForTrackIDCallback;
    DidProvideContentKeyRequestInitializationDataForTrackIDCallback m_didProvideContentKeyRequestInitializationDataForTrackIDCallback;
    DidProvideContentKeyRequestIdentifierForTrackIDCallback m_didProvideContentKeyRequestIdentifierForTrackIDCallback;
    DidUpdateFormatDescriptionForTrackIDCallback m_didUpdateFormatDescriptionForTrackIDCallback;
};

}

#endif // ENABLE(MEDIA_SOURCE)
