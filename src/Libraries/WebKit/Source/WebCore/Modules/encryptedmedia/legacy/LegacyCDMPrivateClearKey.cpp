/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 26, 2024.
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
#include "LegacyCDMPrivateClearKey.h"

#if ENABLE(LEGACY_ENCRYPTED_MEDIA)

#include "LegacyCDM.h"
#include "LegacyCDMSessionClearKey.h"
#include "ContentType.h"
#include "MediaPlayer.h"
#include "PlatformMediaResourceLoader.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(LegacyCDMPrivateClearKey);

bool LegacyCDMPrivateClearKey::supportsKeySystem(const String& keySystem)
{
    if (!equalLettersIgnoringASCIICase(keySystem, "org.w3c.clearkey"_s))
        return false;

    // The MediaPlayer must also support the key system:
    return MediaPlayer::supportsKeySystem(keySystem, emptyString());
}

bool LegacyCDMPrivateClearKey::supportsKeySystemAndMimeType(const String& keySystem, const String& mimeType)
{
    if (!equalLettersIgnoringASCIICase(keySystem, "org.w3c.clearkey"_s))
        return false;

    // The MediaPlayer must also support the key system:
    return MediaPlayer::supportsKeySystem(keySystem, mimeType);
}

bool LegacyCDMPrivateClearKey::supportsMIMEType(const String& mimeType) const
{
    return MediaPlayer::supportsKeySystem(m_cdm->keySystem(), mimeType);
}

RefPtr<LegacyCDMSession> LegacyCDMPrivateClearKey::createSession(LegacyCDMSessionClient& client)
{
    return CDMSessionClearKey::create(client);
}

void LegacyCDMPrivateClearKey::ref() const
{
    m_cdm->ref();
}

void LegacyCDMPrivateClearKey::deref() const
{
    m_cdm->deref();
}

}

#endif
