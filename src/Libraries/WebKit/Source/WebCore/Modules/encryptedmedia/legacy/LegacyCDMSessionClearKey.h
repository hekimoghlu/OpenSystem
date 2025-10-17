/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 14, 2023.
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
#include <wtf/RefCounted.h>
#include <wtf/RobinHoodHashMap.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/text/WTFString.h>

#if ENABLE(LEGACY_ENCRYPTED_MEDIA)

namespace WebCore {

class CDMSessionClearKey final : public LegacyCDMSession, public RefCounted<CDMSessionClearKey> {
    WTF_MAKE_TZONE_ALLOCATED(CDMSessionClearKey);
public:
    static Ref<CDMSessionClearKey> create(LegacyCDMSessionClient& client)
    {
        return adoptRef(*new CDMSessionClearKey(client));
    }

    virtual ~CDMSessionClearKey();

    void ref() const final { RefCounted::ref(); }
    void deref() const final { RefCounted::deref(); }

    // CDMSessionPrivate
    LegacyCDMSessionType type() override { return CDMSessionTypeClearKey; }
    const String& sessionId() const override { return m_sessionId; }
    RefPtr<Uint8Array> generateKeyRequest(const String& mimeType, Uint8Array*, String&, unsigned short&, uint32_t&) override;
    void releaseKeys() override;
    bool update(Uint8Array*, RefPtr<Uint8Array>&, unsigned short&, uint32_t&) override;
    RefPtr<ArrayBuffer> cachedKeyForKeyID(const String&) const override;

private:
    CDMSessionClearKey(LegacyCDMSessionClient&);

    WeakPtr<LegacyCDMSessionClient> m_client;
    RefPtr<Uint8Array> m_initData;
    MemoryCompactRobinHoodHashMap<String, Vector<uint8_t>> m_cachedKeys;
    String m_sessionId;
};

} // namespace WebCore

#endif // ENABLE(LEGACY_ENCRYPTED_MEDIA)
