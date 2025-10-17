/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 5, 2022.
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
#include "WebLockRegistryProxy.h"

#include "Connection.h"
#include "RemoteWebLockRegistryMessages.h"
#include "WebLockRegistryProxyMessages.h"
#include "WebProcessProxy.h"
#include "WebsiteDataStore.h"
#include <WebCore/WebLock.h>
#include <WebCore/WebLockIdentifier.h>
#include <WebCore/WebLockManagerSnapshot.h>
#include <WebCore/WebLockRegistry.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebKit {

#define MESSAGE_CHECK(assertion) MESSAGE_CHECK_BASE(assertion, m_process->connection())
#define MESSAGE_CHECK_COMPLETION(assertion, completion) MESSAGE_CHECK_COMPLETION_BASE(assertion, m_process->connection(), completion)

WTF_MAKE_TZONE_ALLOCATED_IMPL(WebLockRegistryProxy);

WebLockRegistryProxy::WebLockRegistryProxy(WebProcessProxy& process)
    : m_process(process)
{
    protectedProcess()->addMessageReceiver(Messages::WebLockRegistryProxy::messageReceiverName(), *this);
}

WebLockRegistryProxy::~WebLockRegistryProxy()
{
    protectedProcess()->removeMessageReceiver(Messages::WebLockRegistryProxy::messageReceiverName());
}

void WebLockRegistryProxy::requestLock(WebCore::ClientOrigin&& clientOrigin, WebCore::WebLockIdentifier lockIdentifier, WebCore::ScriptExecutionContextIdentifier clientID, String&& name, WebCore::WebLockMode lockMode, bool steal, bool ifAvailable)
{
    Ref process = m_process.get();
    MESSAGE_CHECK(lockIdentifier.processIdentifier() == process->coreProcessIdentifier());
    MESSAGE_CHECK(clientID.processIdentifier() == process->coreProcessIdentifier());
    MESSAGE_CHECK(name.length() <= WebCore::WebLock::maxNameLength);
    m_hasEverRequestedLocks = true;

    RefPtr dataStore = process->websiteDataStore();
    if (!dataStore) {
        process->send(Messages::RemoteWebLockRegistry::DidCompleteLockRequest(lockIdentifier, clientID, false), 0);
        return;
    }

    dataStore->protectedWebLockRegistry()->requestLock(process->sessionID(), WTFMove(clientOrigin), lockIdentifier, clientID, WTFMove(name), lockMode, steal, ifAvailable, [weakThis = WeakPtr { *this }, lockIdentifier, clientID](bool success) {
        if (weakThis)
            weakThis->protectedProcess()->send(Messages::RemoteWebLockRegistry::DidCompleteLockRequest(lockIdentifier, clientID, success), 0);
    }, [weakThis = WeakPtr { *this }, lockIdentifier, clientID] {
        if (weakThis)
            weakThis->protectedProcess()->send(Messages::RemoteWebLockRegistry::DidStealLock(lockIdentifier, clientID), 0);
    });
}

void WebLockRegistryProxy::releaseLock(WebCore::ClientOrigin&& clientOrigin, WebCore::WebLockIdentifier lockIdentifier, WebCore::ScriptExecutionContextIdentifier clientID, String&& name)
{
    MESSAGE_CHECK(lockIdentifier.processIdentifier() == m_process->coreProcessIdentifier());
    MESSAGE_CHECK(clientID.processIdentifier() == m_process->coreProcessIdentifier());
    Ref process = m_process.get();
    if (RefPtr dataStore = process->websiteDataStore())
        dataStore->protectedWebLockRegistry()->releaseLock(process->sessionID(), WTFMove(clientOrigin), lockIdentifier, clientID, WTFMove(name));
}

void WebLockRegistryProxy::abortLockRequest(WebCore::ClientOrigin&& clientOrigin, WebCore::WebLockIdentifier lockIdentifier, WebCore::ScriptExecutionContextIdentifier clientID, String&& name, CompletionHandler<void(bool)>&& completionHandler)
{
    MESSAGE_CHECK_COMPLETION(lockIdentifier.processIdentifier() == m_process->coreProcessIdentifier(), completionHandler(false));
    MESSAGE_CHECK_COMPLETION(clientID.processIdentifier() == m_process->coreProcessIdentifier(), completionHandler(false));
    RefPtr dataStore = m_process->websiteDataStore();
    if (!dataStore) {
        completionHandler(false);
        return;
    }

    dataStore->protectedWebLockRegistry()->abortLockRequest(protectedProcess()->sessionID(), WTFMove(clientOrigin), lockIdentifier, clientID, WTFMove(name), WTFMove(completionHandler));
}

void WebLockRegistryProxy::snapshot(WebCore::ClientOrigin&& clientOrigin, CompletionHandler<void(WebCore::WebLockManagerSnapshot&&)>&& completionHandler)
{
    RefPtr dataStore = m_process->websiteDataStore();
    if (!dataStore) {
        completionHandler(WebCore::WebLockManagerSnapshot { });
        return;
    }

    dataStore->protectedWebLockRegistry()->snapshot(protectedProcess()->sessionID(), WTFMove(clientOrigin), WTFMove(completionHandler));
}

void WebLockRegistryProxy::clientIsGoingAway(WebCore::ClientOrigin&& clientOrigin, WebCore::ScriptExecutionContextIdentifier clientID)
{
    Ref process = m_process.get();
    MESSAGE_CHECK(clientID.processIdentifier() == process->coreProcessIdentifier());
    if (RefPtr dataStore = WebsiteDataStore::existingDataStoreForSessionID(process->sessionID()))
        dataStore->protectedWebLockRegistry()->clientIsGoingAway(process->sessionID(), WTFMove(clientOrigin), clientID);
}

void WebLockRegistryProxy::processDidExit()
{
    if (!m_hasEverRequestedLocks)
        return;

    Ref process = m_process.get();
    if (RefPtr dataStore = WebsiteDataStore::existingDataStoreForSessionID(process->sessionID()))
        dataStore->protectedWebLockRegistry()->clientsAreGoingAway(process->coreProcessIdentifier());
}

#undef MESSAGE_CHECK
#undef MESSAGE_CHECK_COMPLETION

} // namespace WebKit
