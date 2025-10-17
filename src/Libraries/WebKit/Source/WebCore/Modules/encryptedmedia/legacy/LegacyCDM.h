/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 29, 2023.
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

#if ENABLE(LEGACY_ENCRYPTED_MEDIA)

#include "LegacyCDMSession.h"
#include <wtf/CheckedPtr.h>
#include <wtf/Forward.h>
#include <wtf/Function.h>
#include <wtf/RefCountedAndCanMakeWeakPtr.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

class LegacyCDM;
class CDMPrivateInterface;
class MediaPlayer;

using CreateCDM = Function<std::unique_ptr<CDMPrivateInterface>(LegacyCDM&)>;
using CDMSupportsKeySystem = Function<bool(const String&)>;
using CDMSupportsKeySystemAndMimeType = Function<bool(const String&, const String&)>;

class LegacyCDMClient : public CanMakeCheckedPtr<LegacyCDMClient> {
    WTF_MAKE_FAST_ALLOCATED;
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(LegacyCDMClient);
public:
    virtual ~LegacyCDMClient() = default;

    virtual RefPtr<MediaPlayer> cdmMediaPlayer(const LegacyCDM*) const = 0;
};

class WEBCORE_EXPORT LegacyCDM final : public RefCountedAndCanMakeWeakPtr<LegacyCDM> {
    WTF_MAKE_TZONE_ALLOCATED_EXPORT(LegacyCDM, WEBCORE_EXPORT);
public:
    enum CDMErrorCode { NoError, UnknownError, ClientError, ServiceError, OutputError, HardwareChangeError, DomainError };
    static bool supportsKeySystem(const String&);
    static bool keySystemSupportsMimeType(const String& keySystem, const String& mimeType);
    static RefPtr<LegacyCDM> create(const String& keySystem);
    static void registerCDMFactory(CreateCDM&&, CDMSupportsKeySystem&&, CDMSupportsKeySystemAndMimeType&&);
    ~LegacyCDM();

    static void resetFactories();
    static void clearFactories();

    bool supportsMIMEType(const String&) const;
    RefPtr<LegacyCDMSession> createSession(LegacyCDMSessionClient&);

    const String& keySystem() const { return m_keySystem; }

    LegacyCDMClient* client() const { return m_client.get(); }
    void setClient(LegacyCDMClient* client) { m_client = client; }

    RefPtr<MediaPlayer> mediaPlayer() const;
    CDMPrivateInterface* cdmPrivate() const { return m_private.get(); }
    RefPtr<CDMPrivateInterface> protectedCDMPrivate() const;

private:
    explicit LegacyCDM(const String& keySystem);

    String m_keySystem;
    CheckedPtr<LegacyCDMClient> m_client;
    std::unique_ptr<CDMPrivateInterface> m_private;
};

}

#endif // ENABLE(LEGACY_ENCRYPTED_MEDIA)
