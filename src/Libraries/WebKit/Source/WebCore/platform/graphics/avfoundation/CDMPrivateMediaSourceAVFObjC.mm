/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 2, 2021.
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
#import "config.h"
#import "CDMPrivateMediaSourceAVFObjC.h"

#if ENABLE(LEGACY_ENCRYPTED_MEDIA) && ENABLE(MEDIA_SOURCE)

#import "CDMSessionAVContentKeySession.h"
#import "ContentType.h"
#import "LegacyCDM.h"
#import "MediaPlayerPrivateMediaSourceAVFObjC.h"
#import <JavaScriptCore/RegularExpression.h>
#import <wtf/NeverDestroyed.h>
#import <wtf/TZoneMallocInlines.h>
#import <wtf/text/StringToIntegerConversion.h>
#import <wtf/text/StringView.h>

#import "VideoToolboxSoftLink.h"

using JSC::Yarr::RegularExpression;

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(CDMPrivateMediaSourceAVFObjC);

auto CDMPrivateMediaSourceAVFObjC::parseKeySystem(const String& keySystem) -> std::optional<KeySystemParameters>
{
    static NeverDestroyed<RegularExpression> keySystemRE("^com\\.apple\\.fps\\.[23]_\\d+(?:,\\d+)*$"_s, OptionSet<JSC::Yarr::Flags> { JSC::Yarr::Flags::IgnoreCase });

    if (keySystemRE.get().match(keySystem) < 0)
        return std::nullopt;

    StringView keySystemView { keySystem };

    int cdmVersion = parseInteger<int>(keySystemView.substring(14, 1)).value();
    Vector<int> protocolVersions;
    for (auto protocolVersionString : keySystemView.substring(16).split(','))
        protocolVersions.append(parseInteger<int>(protocolVersionString).value());
    return { { cdmVersion, WTFMove(protocolVersions) } };
}

CDMPrivateMediaSourceAVFObjC::~CDMPrivateMediaSourceAVFObjC()
{
    for (auto& session : m_sessions)
        session->invalidateCDM();
}

static bool queryDecoderAvailability()
{
    if (!canLoad_VideoToolbox_VTGetGVADecoderAvailability()) {
        return true;
    }
    uint32_t totalInstanceCount = 0;
    OSStatus status = VTGetGVADecoderAvailability(&totalInstanceCount, nullptr);
    return status == noErr && totalInstanceCount;
}

bool CDMPrivateMediaSourceAVFObjC::supportsKeySystem(const String& keySystem)
{
    if (!queryDecoderAvailability())
        return false;

    auto parameters = parseKeySystem(keySystem);
    if (!parameters)
        return false;

    if (parameters.value().version == 3 && !CDMSessionAVContentKeySession::isAvailable())
        return false;

    return true;
}

bool CDMPrivateMediaSourceAVFObjC::supportsKeySystemAndMimeType(const String& keySystem, const String& mimeType)
{
    if (!supportsKeySystem(keySystem))
        return false;

    if (mimeType.isEmpty())
        return true;

    // FIXME: Why is this ignoring case since the check in supportsMIMEType is checking case?
    if (equalLettersIgnoringASCIICase(mimeType, "keyrelease"_s))
        return true;

    MediaEngineSupportParameters parameters;
    parameters.isMediaSource = true;
    parameters.type = ContentType(mimeType);

    return MediaPlayerPrivateMediaSourceAVFObjC::supportsTypeAndCodecs(parameters) != MediaPlayer::SupportsType::IsNotSupported;
}

bool CDMPrivateMediaSourceAVFObjC::supportsMIMEType(const String& mimeType) const
{
    // FIXME: Why is this checking case since the check in supportsKeySystemAndMimeType is ignoring case?
    if (mimeType == "keyrelease"_s)
        return true;

    MediaEngineSupportParameters parameters;
    parameters.isMediaSource = true;
    parameters.type = ContentType(mimeType);

    return MediaPlayerPrivateMediaSourceAVFObjC::supportsTypeAndCodecs(parameters) != MediaPlayer::SupportsType::IsNotSupported;
}

RefPtr<LegacyCDMSession> CDMPrivateMediaSourceAVFObjC::createSession(LegacyCDMSessionClient& client)
{
    String keySystem = m_cdm->keySystem(); // Local copy for StringView usage
    auto parameters = parseKeySystem(m_cdm->keySystem());
    ASSERT(parameters);
    if (!parameters)
        return nullptr;

    RefPtr session = CDMSessionAVContentKeySession::create(WTFMove(parameters.value().protocols), parameters.value().version, *this, client);

    m_sessions.append(session.get());
    return WTFMove(session);
}

void CDMPrivateMediaSourceAVFObjC::invalidateSession(CDMSessionMediaSourceAVFObjC* session)
{
    ASSERT(m_sessions.contains(session));
    m_sessions.removeAll(session);
}

void CDMPrivateMediaSourceAVFObjC::ref() const
{
    m_cdm->ref();
}

void CDMPrivateMediaSourceAVFObjC::deref() const
{
    m_cdm->deref();
}

}

#endif // ENABLE(LEGACY_ENCRYPTED_MEDIA) && ENABLE(MEDIA_SOURCE)
