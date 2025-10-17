/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 29, 2022.
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

#if ENABLE(LEGACY_ENCRYPTED_MEDIA)

#include "LegacyCDM.h"

#include "LegacyCDMPrivateClearKey.h"
#include "LegacyCDMPrivateMediaPlayer.h"
#include "LegacyCDMSession.h"
#include "MediaPlayer.h"
#include "WebKitMediaKeys.h"
#include <wtf/NeverDestroyed.h>
#include <wtf/TZoneMallocInlines.h>
#include <wtf/text/WTFString.h>

#if HAVE(AVCONTENTKEYSESSION) && ENABLE(MEDIA_SOURCE)
#include "CDMPrivateMediaSourceAVFObjC.h"
#endif

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(CDMPrivateInterface);

struct LegacyCDMFactory {
    CreateCDM constructor;
    CDMSupportsKeySystem supportsKeySystem;
    CDMSupportsKeySystemAndMimeType supportsKeySystemAndMimeType;
};

static void platformRegisterFactories(Vector<LegacyCDMFactory>& factories)
{
    factories.append({ [](LegacyCDM& cdm) { return makeUniqueWithoutRefCountedCheck<LegacyCDMPrivateClearKey>(cdm); }, LegacyCDMPrivateClearKey::supportsKeySystem, LegacyCDMPrivateClearKey::supportsKeySystemAndMimeType });
    // FIXME: initialize specific UA CDMs. http://webkit.org/b/109318, http://webkit.org/b/109320
    factories.append({ [](LegacyCDM& cdm) { return makeUniqueWithoutRefCountedCheck<CDMPrivateMediaPlayer>(cdm); }, CDMPrivateMediaPlayer::supportsKeySystem, CDMPrivateMediaPlayer::supportsKeySystemAndMimeType });
#if HAVE(AVCONTENTKEYSESSION) && ENABLE(MEDIA_SOURCE)
    factories.append({ [](LegacyCDM& cdm) { return makeUniqueWithoutRefCountedCheck<CDMPrivateMediaSourceAVFObjC>(cdm); }, CDMPrivateMediaSourceAVFObjC::supportsKeySystem, CDMPrivateMediaSourceAVFObjC::supportsKeySystemAndMimeType });
#endif
}

static Vector<LegacyCDMFactory>& installedCDMFactories()
{
    static NeverDestroyed<Vector<LegacyCDMFactory>> cdms;
    static std::once_flag registerDefaults;
    std::call_once(registerDefaults, [&] {
        platformRegisterFactories(cdms);
    });
    return cdms;
}

WTF_MAKE_TZONE_ALLOCATED_IMPL(LegacyCDM);

void LegacyCDM::resetFactories()
{
    clearFactories();
    platformRegisterFactories(installedCDMFactories());
}

void LegacyCDM::clearFactories()
{
    installedCDMFactories().clear();
}

void LegacyCDM::registerCDMFactory(CreateCDM&& constructor, CDMSupportsKeySystem&& supportsKeySystem, CDMSupportsKeySystemAndMimeType&& supportsKeySystemAndMimeType)
{
    installedCDMFactories().append({ WTFMove(constructor), WTFMove(supportsKeySystem), WTFMove(supportsKeySystemAndMimeType) });
}

static LegacyCDMFactory* CDMFactoryForKeySystem(const String& keySystem)
{
    for (auto& factory : installedCDMFactories()) {
        if (factory.supportsKeySystem(keySystem))
            return &factory;
    }
    return nullptr;
}

bool LegacyCDM::supportsKeySystem(const String& keySystem)
{
    return CDMFactoryForKeySystem(keySystem);
}

bool LegacyCDM::keySystemSupportsMimeType(const String& keySystem, const String& mimeType)
{
    if (LegacyCDMFactory* factory = CDMFactoryForKeySystem(keySystem))
        return factory->supportsKeySystemAndMimeType(keySystem, mimeType);
    return false;
}

RefPtr<LegacyCDM> LegacyCDM::create(const String& keySystem)
{
    if (!supportsKeySystem(keySystem))
        return nullptr;

    return adoptRef(*new LegacyCDM(keySystem));
}

LegacyCDM::LegacyCDM(const String& keySystem)
    : m_keySystem(keySystem)
    , m_private(CDMFactoryForKeySystem(keySystem)->constructor(*this))
{
}

LegacyCDM::~LegacyCDM() = default;

bool LegacyCDM::supportsMIMEType(const String& mimeType) const
{
    return protectedCDMPrivate()->supportsMIMEType(mimeType);
}

RefPtr<CDMPrivateInterface> LegacyCDM::protectedCDMPrivate() const
{
    return cdmPrivate();
}

RefPtr<LegacyCDMSession> LegacyCDM::createSession(LegacyCDMSessionClient& client)
{
    RefPtr session = protectedCDMPrivate()->createSession(client);
    if (mediaPlayer())
        mediaPlayer()->setCDMSession(session.get());
    return session;
}

RefPtr<MediaPlayer> LegacyCDM::mediaPlayer() const
{
    if (!m_client)
        return nullptr;
    return m_client->cdmMediaPlayer(this);
}

}

#endif // ENABLE(LEGACY_ENCRYPTED_MEDIA)
