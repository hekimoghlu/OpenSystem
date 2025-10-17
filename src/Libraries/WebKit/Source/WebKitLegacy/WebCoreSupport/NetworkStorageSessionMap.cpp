/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 19, 2023.
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
#include "NetworkStorageSessionMap.h"

#include <WebCore/NetworkStorageSession.h>
#include <pal/SessionID.h>
#include <wtf/MainThread.h>
#include <wtf/ProcessID.h>
#include <wtf/ProcessPrivilege.h>
#include <wtf/UUID.h>
#include <wtf/text/MakeString.h>

static std::unique_ptr<WebCore::NetworkStorageSession>& defaultNetworkStorageSession()
{
    ASSERT(isMainThread());
    static NeverDestroyed<std::unique_ptr<WebCore::NetworkStorageSession>> session;
    return session;
}

static HashMap<PAL::SessionID, std::unique_ptr<WebCore::NetworkStorageSession>>& globalSessionMap()
{
    static NeverDestroyed<HashMap<PAL::SessionID, std::unique_ptr<WebCore::NetworkStorageSession>>> map;
    return map;
}

WebCore::NetworkStorageSession* NetworkStorageSessionMap::storageSession(PAL::SessionID sessionID)
{
    if (sessionID == PAL::SessionID::defaultSessionID())
        return &defaultStorageSession();
    return globalSessionMap().get(sessionID);
}

WebCore::NetworkStorageSession& NetworkStorageSessionMap::defaultStorageSession()
{
    if (!defaultNetworkStorageSession())
        defaultNetworkStorageSession() = makeUnique<WebCore::NetworkStorageSession>(PAL::SessionID::defaultSessionID());
    return *defaultNetworkStorageSession();
}

void NetworkStorageSessionMap::switchToNewTestingSession()
{
#if PLATFORM(COCOA)
    // Session name should be short enough for shared memory region name to be under the limit, otherwise sandbox rules won't work (see <rdar://problem/13642852>).
    auto session = WebCore::createPrivateStorageSession(makeString("WebKit Test-"_s, getCurrentProcessID()).createCFString().get());

    RetainPtr<CFHTTPCookieStorageRef> cookieStorage;
    if (WebCore::NetworkStorageSession::processMayUseCookieAPI()) {
        ASSERT(hasProcessPrivilege(ProcessPrivilege::CanAccessRawCookies));
        if (session)
            cookieStorage = adoptCF(_CFURLStorageSessionCopyCookieStorage(kCFAllocatorDefault, session.get()));
    }

    defaultNetworkStorageSession() = makeUnique<WebCore::NetworkStorageSession>(PAL::SessionID::defaultSessionID(), WTFMove(session), WTFMove(cookieStorage));
#endif
}

void NetworkStorageSessionMap::ensureSession(PAL::SessionID sessionID, const String& identifierBase)
{
#if PLATFORM(COCOA)
    auto addResult = globalSessionMap().add(sessionID, nullptr);
    if (!addResult.isNewEntry)
        return;

    auto identifier = makeString(identifierBase, ".PrivateBrowsing."_s, WTF::UUID::createVersion4()).createCFString();

    RetainPtr<CFURLStorageSessionRef> storageSession;
    if (sessionID.isEphemeral())
        storageSession = WebCore::createPrivateStorageSession(identifier.get());
    else
        storageSession = WebCore::NetworkStorageSession::createCFStorageSessionForIdentifier(identifier.get());

    RetainPtr<CFHTTPCookieStorageRef> cookieStorage;
    if (WebCore::NetworkStorageSession::processMayUseCookieAPI()) {
        ASSERT(hasProcessPrivilege(ProcessPrivilege::CanAccessRawCookies));
        if (storageSession)
            cookieStorage = adoptCF(_CFURLStorageSessionCopyCookieStorage(kCFAllocatorDefault, storageSession.get()));
    }

    addResult.iterator->value = makeUnique<WebCore::NetworkStorageSession>(sessionID, WTFMove(storageSession), WTFMove(cookieStorage));
#endif
}

void NetworkStorageSessionMap::destroySession(PAL::SessionID sessionID)
{
    globalSessionMap().remove(sessionID);
}
