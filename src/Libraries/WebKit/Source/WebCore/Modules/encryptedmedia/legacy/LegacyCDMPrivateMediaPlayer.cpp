/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 10, 2023.
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
#include "LegacyCDMPrivateMediaPlayer.h"

#if ENABLE(LEGACY_ENCRYPTED_MEDIA)

#include "LegacyCDM.h"
#include "LegacyCDMSession.h"
#include "ContentType.h"
#include "MediaPlayer.h"
#include <wtf/TZoneMallocInlines.h>

#if PLATFORM(IOS_FAMILY)
#include <wtf/SoftLinking.h>
#endif

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(CDMPrivateMediaPlayer);

bool CDMPrivateMediaPlayer::supportsKeySystem(const String& keySystem)
{
    return MediaPlayer::supportsKeySystem(keySystem, emptyString());
}

bool CDMPrivateMediaPlayer::supportsKeySystemAndMimeType(const String& keySystem, const String& mimeType)
{
    return MediaPlayer::supportsKeySystem(keySystem, mimeType);
}

bool CDMPrivateMediaPlayer::supportsMIMEType(const String& mimeType) const
{
    return MediaPlayer::supportsKeySystem(m_cdm->keySystem(), mimeType);
}

RefPtr<LegacyCDMSession> CDMPrivateMediaPlayer::createSession(LegacyCDMSessionClient& client)
{
    Ref cdm = m_cdm.get();
    auto mediaPlayer = cdm->mediaPlayer();
    if (!mediaPlayer)
        return nullptr;

    return mediaPlayer->createSession(cdm->keySystem(), client);
}

void CDMPrivateMediaPlayer::ref() const
{
    m_cdm->ref();
}

void CDMPrivateMediaPlayer::deref() const
{
    m_cdm->deref();
}

}

#endif
