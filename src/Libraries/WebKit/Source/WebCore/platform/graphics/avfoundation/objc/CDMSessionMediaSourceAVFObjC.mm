/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 11, 2022.
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
#import "CDMSessionMediaSourceAVFObjC.h"

#if ENABLE(LEGACY_ENCRYPTED_MEDIA) && ENABLE(MEDIA_SOURCE)

#import "CDMPrivateMediaSourceAVFObjC.h"
#import "Logging.h"
#import "WebCoreNSErrorExtras.h"
#import <AVFoundation/AVError.h>
#import <wtf/FileSystem.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(CDMSessionMediaSourceAVFObjC);

CDMSessionMediaSourceAVFObjC::CDMSessionMediaSourceAVFObjC(CDMPrivateMediaSourceAVFObjC& cdm, LegacyCDMSessionClient& client)
    : m_cdm(cdm)
    , m_client(client)
#if !RELEASE_LOG_DISABLED
    , m_logger(client.logger())
    , m_logIdentifier(client.logIdentifier())
#endif
{
}

CDMSessionMediaSourceAVFObjC::~CDMSessionMediaSourceAVFObjC()
{
    if (RefPtr cdm = m_cdm.get())
        cdm->invalidateSession(this);

    for (auto& sourceBuffer : m_sourceBuffers)
        sourceBuffer->unregisterForErrorNotifications(this);
    m_sourceBuffers.clear();
}

void CDMSessionMediaSourceAVFObjC::videoRendererDidReceiveError(WebSampleBufferVideoRendering *, NSError *error, bool& shouldIgnore)
{
    if (!m_client)
        return;

    unsigned long code = mediaKeyErrorSystemCode(error);

    // FIXME(142246): Remove the following once <rdar://problem/20027434> is resolved.
    shouldIgnore = m_stopped && code == 12785;
    if (!shouldIgnore)
        m_client->sendError(LegacyCDMSessionClient::MediaKeyErrorDomain, code);
}

void CDMSessionMediaSourceAVFObjC::audioRendererDidReceiveError(AVSampleBufferAudioRenderer *, NSError *error, bool& shouldIgnore)
{
    if (!m_client)
        return;

    unsigned long code = mediaKeyErrorSystemCode(error);

    // FIXME(142246): Remove the following once <rdar://problem/20027434> is resolved.
    shouldIgnore = m_stopped && code == 12785;
    if (!shouldIgnore)
        m_client->sendError(LegacyCDMSessionClient::MediaKeyErrorDomain, code);
}


void CDMSessionMediaSourceAVFObjC::addSourceBuffer(SourceBufferPrivateAVFObjC* sourceBuffer)
{
    ASSERT(!m_sourceBuffers.contains(sourceBuffer));
    ASSERT(sourceBuffer);

    m_sourceBuffers.append(sourceBuffer);
    sourceBuffer->registerForErrorNotifications(this);
}

void CDMSessionMediaSourceAVFObjC::removeSourceBuffer(SourceBufferPrivateAVFObjC* sourceBuffer)
{
    ASSERT(m_sourceBuffers.contains(sourceBuffer));
    ASSERT(sourceBuffer);

    sourceBuffer->unregisterForErrorNotifications(this);
    m_sourceBuffers.remove(m_sourceBuffers.find(sourceBuffer));
}

String CDMSessionMediaSourceAVFObjC::storagePath() const
{
    if (!m_client)
        return emptyString();

    String storageDirectory = m_client->mediaKeysStorageDirectory();
    if (storageDirectory.isEmpty())
        return emptyString();

    return FileSystem::pathByAppendingComponent(storageDirectory, "SecureStop.plist"_s);
}

#if !RELEASE_LOG_DISABLED
WTFLogChannel& CDMSessionMediaSourceAVFObjC::logChannel() const
{
    return LogEME;
}
#endif

}

#endif // ENABLE(LEGACY_ENCRYPTED_MEDIA) && ENABLE(MEDIA_SOURCE)
