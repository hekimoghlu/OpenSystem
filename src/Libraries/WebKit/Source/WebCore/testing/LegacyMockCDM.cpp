/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 30, 2024.
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
#include "LegacyMockCDM.h"

#if ENABLE(LEGACY_ENCRYPTED_MEDIA)

#include "LegacyCDM.h"
#include "LegacyCDMSession.h"
#include "WebKitMediaKeyError.h"
#include <JavaScriptCore/JSCInlines.h>
#include <JavaScriptCore/TypedArrayInlines.h>
#include <JavaScriptCore/Uint8Array.h>
#include <wtf/RefCounted.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(LegacyMockCDM);

class MockCDMSession : public LegacyCDMSession, public RefCounted<MockCDMSession> {
    WTF_MAKE_TZONE_ALLOCATED_INLINE(MockCDMSession);
public:
    static Ref<MockCDMSession> create(LegacyCDMSessionClient& client)
    {
        return adoptRef(*new MockCDMSession(client));
    }

    virtual ~MockCDMSession() = default;

    void ref() const final { RefCounted::ref(); }
    void deref() const final { RefCounted::deref(); }

    const String& sessionId() const override { return m_sessionId; }
    RefPtr<Uint8Array> generateKeyRequest(const String& mimeType, Uint8Array* initData, String& destinationURL, unsigned short& errorCode, uint32_t& systemCode) override;
    void releaseKeys() override;
    bool update(Uint8Array*, RefPtr<Uint8Array>& nextMessage, unsigned short& errorCode, uint32_t& systemCode) override;
    RefPtr<ArrayBuffer> cachedKeyForKeyID(const String&) const override { return nullptr; }

protected:
    MockCDMSession(LegacyCDMSessionClient&);

    WeakPtr<LegacyCDMSessionClient> m_client;
    String m_sessionId;
};

bool LegacyMockCDM::supportsKeySystem(const String& keySystem)
{
    return equalLettersIgnoringASCIICase(keySystem, "com.webcore.mock"_s);
}

bool LegacyMockCDM::supportsKeySystemAndMimeType(const String& keySystem, const String& mimeType)
{
    if (!supportsKeySystem(keySystem))
        return false;

    return equalLettersIgnoringASCIICase(mimeType, "video/mock"_s);
}

bool LegacyMockCDM::supportsMIMEType(const String& mimeType) const
{
    return equalLettersIgnoringASCIICase(mimeType, "video/mock"_s);
}

RefPtr<LegacyCDMSession> LegacyMockCDM::createSession(LegacyCDMSessionClient& client)
{
    return MockCDMSession::create(client);
}

void LegacyMockCDM::ref() const
{
    m_cdm->ref();
}

void LegacyMockCDM::deref() const
{
    m_cdm->deref();
}

static Uint8Array* initDataPrefix()
{
    const unsigned char prefixData[] = { 'm', 'o', 'c', 'k' };
    static Uint8Array& prefix { Uint8Array::create(prefixData, std::size(prefixData)).leakRef() };

    return &prefix;
}

static Uint8Array* keyPrefix()
{
    static const unsigned char prefixData[] = {'k', 'e', 'y'};
    static Uint8Array& prefix { Uint8Array::create(prefixData, std::size(prefixData)).leakRef() };

    return &prefix;
}

static Uint8Array* keyRequest()
{
    static const unsigned char requestData[] = {'r', 'e', 'q', 'u', 'e', 's', 't'};
    static Uint8Array& request { Uint8Array::create(requestData, std::size(requestData)).leakRef() };

    return &request;
}

static String generateSessionId()
{
    static int monotonicallyIncreasingSessionId = 0;
    return String::number(monotonicallyIncreasingSessionId++);
}

MockCDMSession::MockCDMSession(LegacyCDMSessionClient& client)
    : m_client(client)
    , m_sessionId(generateSessionId())
{
}

RefPtr<Uint8Array> MockCDMSession::generateKeyRequest(const String&, Uint8Array* initData, String&, unsigned short& errorCode, uint32_t&)
{
    for (unsigned i = 0; i < initDataPrefix()->length(); ++i) {
        if (!initData || i >= initData->length() || initData->item(i) != initDataPrefix()->item(i)) {
            errorCode = WebKitMediaKeyError::MEDIA_KEYERR_UNKNOWN;
            return nullptr;
        }
    }
    return keyRequest();
}

void MockCDMSession::releaseKeys()
{
    // no-op
}

bool MockCDMSession::update(Uint8Array* key, RefPtr<Uint8Array>&, unsigned short& errorCode, uint32_t&)
{
    for (unsigned i = 0; i < keyPrefix()->length(); ++i) {
        if (i >= key->length() || key->item(i) != keyPrefix()->item(i)) {
            errorCode = WebKitMediaKeyError::MEDIA_KEYERR_CLIENT;
            return false;
        }
    }
    return true;
}

}

#endif // ENABLE(LEGACY_ENCRYPTED_MEDIA)
