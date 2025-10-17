/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 2, 2023.
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

#include "Logging.h"
#include "SourceBufferParser.h"
#include <wtf/Box.h>
#include <wtf/LoggerHelper.h>
#include <wtf/TypeCasts.h>
#include <wtf/Vector.h>
#include <wtf/WeakPtr.h>

OBJC_CLASS AVAsset;
OBJC_CLASS AVStreamDataParser;
OBJC_CLASS NSData;
OBJC_CLASS NSError;
OBJC_CLASS WebAVStreamDataParserListener;

typedef struct opaqueCMSampleBuffer *CMSampleBufferRef;

namespace WebCore {

class SourceBufferParserAVFObjC final
    : public SourceBufferParser
    , public CanMakeWeakPtr<SourceBufferParserAVFObjC>
    , private LoggerHelper {
public:
    static MediaPlayerEnums::SupportsType isContentTypeSupported(const ContentType&);

    SourceBufferParserAVFObjC();
    virtual ~SourceBufferParserAVFObjC();

    AVStreamDataParser* streamDataParser() const { return m_parser.get(); }

    Type type() const { return Type::AVFObjC; }
    Expected<void, PlatformMediaError> appendData(Segment&&, AppendFlags = AppendFlags::None) final;
    void flushPendingMediaData() final;
    void resetParserState() final;
    void invalidate() final;
#if !RELEASE_LOG_DISABLED
    void setLogger(const Logger&, uint64_t identifier) final;
#endif

    void didParseStreamDataAsAsset(AVAsset*);
    void didFailToParseStreamDataWithError(NSError*);
    void didProvideMediaDataForTrackID(uint64_t trackID, CMSampleBufferRef, const String& mediaType, unsigned flags);
    void willProvideContentKeyRequestInitializationDataForTrackID(uint64_t trackID);
    void didProvideContentKeyRequestInitializationDataForTrackID(NSData*, uint64_t trackID);
    void didProvideContentKeyRequestSpecifierForTrackID(NSData*, uint64_t trackID);

private:
#if !RELEASE_LOG_DISABLED
    const Logger* loggerPtr() const { return m_logger.get(); }
    const Logger& logger() const final { ASSERT(m_logger); return *m_logger.get(); }
    uint64_t logIdentifier() const final { return m_logIdentifier; }
    ASCIILiteral logClassName() const final { return "SourceBufferParserAVFObjC"_s; }
    WTFLogChannel& logChannel() const final { return LogMedia; }
#endif

    RetainPtr<AVStreamDataParser> m_parser;
    RetainPtr<WebAVStreamDataParserListener> m_delegate;
    bool m_parserStateWasReset { false };
    std::optional<int> m_lastErrorCode;

#if !RELEASE_LOG_DISABLED
    RefPtr<const Logger> m_logger;
    uint64_t m_logIdentifier { 0 };
#endif
};
}

SPECIALIZE_TYPE_TRAITS_BEGIN(WebCore::SourceBufferParserAVFObjC)
    static bool isType(const WebCore::SourceBufferParser& parser) { return parser.type() == WebCore::SourceBufferParser::Type::AVFObjC; }
SPECIALIZE_TYPE_TRAITS_END()

#endif // ENABLE(MEDIA_SOURCE)
