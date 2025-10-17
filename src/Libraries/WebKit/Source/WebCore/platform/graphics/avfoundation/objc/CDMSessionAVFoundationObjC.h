/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 1, 2023.
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
#include <wtf/RefCountedAndCanMakeWeakPtr.h>
#include <wtf/RetainPtr.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/WeakPtr.h>
#include <wtf/text/WTFString.h>

#if ENABLE(LEGACY_ENCRYPTED_MEDIA)

OBJC_CLASS AVAssetResourceLoadingRequest;
OBJC_CLASS WebCDMSessionAVFoundationObjCListener;

namespace WebCore {

class MediaPlayerPrivateAVFoundationObjC;

class CDMSessionAVFoundationObjC final : public LegacyCDMSession, public RefCountedAndCanMakeWeakPtr<CDMSessionAVFoundationObjC> {
    WTF_MAKE_TZONE_ALLOCATED(CDMSessionAVFoundationObjC);
public:
    static Ref<CDMSessionAVFoundationObjC> create(MediaPlayerPrivateAVFoundationObjC* parent, LegacyCDMSessionClient& client)
    {
        return adoptRef(*new CDMSessionAVFoundationObjC(parent, client));
    }
    virtual ~CDMSessionAVFoundationObjC();

    void ref() const final { RefCounted::ref(); }
    void deref() const final { RefCounted::deref(); }

    LegacyCDMSessionType type() override { return CDMSessionTypeAVFoundationObjC; }
    const String& sessionId() const override { return m_sessionId; }
    RefPtr<Uint8Array> generateKeyRequest(const String& mimeType, Uint8Array* initData, String& destinationURL, unsigned short& errorCode, uint32_t& systemCode) override;
    void releaseKeys() override;
    bool update(Uint8Array*, RefPtr<Uint8Array>& nextMessage, unsigned short& errorCode, uint32_t& systemCode) override;
    RefPtr<ArrayBuffer> cachedKeyForKeyID(const String&) const override;

    void playerDidReceiveError(NSError *);

private:
    CDMSessionAVFoundationObjC(MediaPlayerPrivateAVFoundationObjC* parent, LegacyCDMSessionClient&);

#if !RELEASE_LOG_DISABLED
    const Logger& logger() const { return m_logger; }
    uint64_t logIdentifier() const { return m_logIdentifier; }
    ASCIILiteral logClassName() const { return "CDMSessionAVFoundationObjC"_s; }
    WTFLogChannel& logChannel() const;
#endif

    ThreadSafeWeakPtr<MediaPlayerPrivateAVFoundationObjC> m_parent;
    WeakPtr<LegacyCDMSessionClient> m_client;
    String m_sessionId;
    RetainPtr<AVAssetResourceLoadingRequest> m_request;

#if !RELEASE_LOG_DISABLED
    Ref<const Logger> m_logger;
    const uint64_t m_logIdentifier;
#endif
};

}

#endif
