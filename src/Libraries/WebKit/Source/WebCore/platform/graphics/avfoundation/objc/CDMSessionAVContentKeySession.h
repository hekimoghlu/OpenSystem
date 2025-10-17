/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 22, 2022.
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

#include "CDMSessionMediaSourceAVFObjC.h"
#include "SourceBufferPrivateAVFObjC.h"
#include <wtf/RefCounted.h>
#include <wtf/RetainPtr.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/WTFSemaphore.h>

#if ENABLE(LEGACY_ENCRYPTED_MEDIA) && ENABLE(MEDIA_SOURCE)

OBJC_CLASS AVContentKeyRequest;
OBJC_CLASS AVContentKeySession;
OBJC_CLASS WebCDMSessionAVContentKeySessionDelegate;

namespace WTF {
class WorkQueue;
}

namespace WebCore {
class CDMSessionAVContentKeySession;
}

namespace WTF {
template<typename T> struct IsDeprecatedWeakRefSmartPointerException;
template<> struct IsDeprecatedWeakRefSmartPointerException<WebCore::CDMSessionAVContentKeySession> : std::true_type { };
}

namespace WebCore {

class CDMPrivateMediaSourceAVFObjC;

class CDMSessionAVContentKeySession : public CDMSessionMediaSourceAVFObjC, public RefCounted<CDMSessionAVContentKeySession> {
    WTF_MAKE_TZONE_ALLOCATED(CDMSessionAVContentKeySession);
public:
    static Ref<CDMSessionAVContentKeySession> create(Vector<int>&& protocolVersions, int cdmVersion, CDMPrivateMediaSourceAVFObjC& parent, LegacyCDMSessionClient& client)
    {
        return adoptRef(*new CDMSessionAVContentKeySession(WTFMove(protocolVersions), cdmVersion, parent, client));
    }

    virtual ~CDMSessionAVContentKeySession();

    void ref() const final { RefCounted::ref(); }
    void deref() const final { RefCounted::deref(); }

    static bool isAvailable();

    // LegacyCDMSession
    LegacyCDMSessionType type() override { return CDMSessionTypeAVContentKeySession; }
    RefPtr<Uint8Array> generateKeyRequest(const String& mimeType, Uint8Array* initData, String& destinationURL, unsigned short& errorCode, uint32_t& systemCode) override;
    void releaseKeys() override;
    bool update(Uint8Array* key, RefPtr<Uint8Array>& nextMessage, unsigned short& errorCode, uint32_t& systemCode) override;
    RefPtr<ArrayBuffer> cachedKeyForKeyID(const String&) const override;

    // CDMSessionMediaSourceAVFObjC
    void addParser(AVStreamDataParser *) override;
    void removeParser(AVStreamDataParser *) override;
    bool isAnyKeyUsable(const Keys&) const override;
    void attachContentKeyToSample(const MediaSampleAVFObjC&) override;

    void didProvideContentKeyRequest(AVContentKeyRequest *);

    bool hasContentKeySession() const { return m_contentKeySession; }
    AVContentKeySession* contentKeySession();

    bool hasContentKeyRequest() const;
    RetainPtr<AVContentKeyRequest> contentKeyRequest() const;

protected:
    CDMSessionAVContentKeySession(Vector<int>&& protocolVersions, int cdmVersion, CDMPrivateMediaSourceAVFObjC&, LegacyCDMSessionClient&);

    RefPtr<Uint8Array> generateKeyReleaseMessage(unsigned short& errorCode, uint32_t& systemCode);

#if !RELEASE_LOG_DISABLED
    ASCIILiteral logClassName() const { return "CDMSessionAVContentKeySession"_s; }
#endif

    RetainPtr<AVContentKeySession> m_contentKeySession;
    RetainPtr<WebCDMSessionAVContentKeySessionDelegate> m_contentKeySessionDelegate;
    Ref<WTF::WorkQueue> m_delegateQueue;
    Semaphore m_hasKeyRequestSemaphore;
    mutable Lock m_keyRequestLock;
    RetainPtr<AVContentKeyRequest> m_keyRequest;
    RefPtr<Uint8Array> m_identifier;
    RefPtr<SharedBuffer> m_initData;
    RetainPtr<NSData> m_expiredSession;
    Vector<int> m_protocolVersions;
    int m_cdmVersion;
    int32_t m_protectedTrackID { 1 };
    enum { Normal, KeyRelease } m_mode;

#if !RELEASE_LOG_DISABLED
    Ref<const Logger> m_logger;
    const uint64_t m_logIdentifier;
#endif
};

inline CDMSessionAVContentKeySession* toCDMSessionAVContentKeySession(LegacyCDMSession* session)
{
    if (!session || session->type() != CDMSessionTypeAVContentKeySession)
        return nullptr;
    return static_cast<CDMSessionAVContentKeySession*>(session);
}

}

#endif
