/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 14, 2022.
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

#include "LegacyCDMSession.h"
#include "SourceBufferPrivateAVFObjC.h"
#include <wtf/AbstractRefCounted.h>
#include <wtf/RetainPtr.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/WeakPtr.h>

#if ENABLE(LEGACY_ENCRYPTED_MEDIA) && ENABLE(MEDIA_SOURCE)

OBJC_CLASS AVStreamDataParser;
OBJC_CLASS NSError;

namespace WebCore {

class CDMPrivateMediaSourceAVFObjC;

class CDMSessionMediaSourceAVFObjC : public LegacyCDMSession, public SourceBufferPrivateAVFObjCErrorClient, public CanMakeWeakPtr<CDMSessionMediaSourceAVFObjC> {
    WTF_MAKE_TZONE_ALLOCATED(CDMSessionMediaSourceAVFObjC);
public:
    CDMSessionMediaSourceAVFObjC(CDMPrivateMediaSourceAVFObjC&, LegacyCDMSessionClient&);
    virtual ~CDMSessionMediaSourceAVFObjC();

    virtual void addParser(AVStreamDataParser*) = 0;
    virtual void removeParser(AVStreamDataParser*) = 0;

    // LegacyCDMSession
    const String& sessionId() const override { return m_sessionId; }

    // SourceBufferPrivateAVFObjCErrorClient
    void videoRendererDidReceiveError(WebSampleBufferVideoRendering *, NSError *, bool& shouldIgnore) override;
ALLOW_NEW_API_WITHOUT_GUARDS_BEGIN
    void audioRendererDidReceiveError(AVSampleBufferAudioRenderer *, NSError *, bool& shouldIgnore) override;
ALLOW_NEW_API_WITHOUT_GUARDS_END

    void addSourceBuffer(SourceBufferPrivateAVFObjC*);
    void removeSourceBuffer(SourceBufferPrivateAVFObjC*);
    void setSessionId(const String& sessionId) { m_sessionId = sessionId; }

    using Keys = Vector<Ref<SharedBuffer>>;
    virtual bool isAnyKeyUsable(const Keys&) const = 0;
    virtual void attachContentKeyToSample(const MediaSampleAVFObjC&) = 0;

    void invalidateCDM() { m_cdm = nullptr; }

protected:
    String storagePath() const;

#if !RELEASE_LOG_DISABLED
    const Logger& logger() const { return m_logger; }
    uint64_t logIdentifier() const { return m_logIdentifier; }
    WTFLogChannel& logChannel() const;
#endif

    WeakPtr<CDMPrivateMediaSourceAVFObjC> m_cdm;
    WeakPtr<LegacyCDMSessionClient> m_client;
    Vector<RefPtr<SourceBufferPrivateAVFObjC>> m_sourceBuffers;
    RefPtr<Uint8Array> m_certificate;
    String m_sessionId;
    bool m_stopped { false };

#if !RELEASE_LOG_DISABLED
    Ref<const Logger> m_logger;
    const uint64_t m_logIdentifier;
#endif
};

inline CDMSessionMediaSourceAVFObjC* toCDMSessionMediaSourceAVFObjC(LegacyCDMSession* session)
{
    if (!session || session->type() != CDMSessionTypeAVContentKeySession)
        return nullptr;
    return static_cast<CDMSessionMediaSourceAVFObjC*>(session);
}

}

#endif // ENABLE(LEGACY_ENCRYPTED_MEDIA) && ENABLE(MEDIA_SOURCE)
