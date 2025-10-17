/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 30, 2025.
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
#include "WebStorageConnection.h"

#include "NetworkProcessConnection.h"
#include "NetworkStorageManagerMessages.h"
#include "WebFileSystemStorageConnection.h"
#include "WebProcess.h"
#include <WebCore/ClientOrigin.h>
#include <WebCore/ExceptionOr.h>
#include <WebCore/FileSystemHandleIdentifier.h>
#include <WebCore/StorageEstimate.h>

namespace WebKit {

Ref<WebStorageConnection> WebStorageConnection::create()
{
    return adoptRef(*new WebStorageConnection);
}

void WebStorageConnection::getPersisted(WebCore::ClientOrigin&& origin, StorageConnection::PersistCallback&& completionHandler)
{
    Ref { connection() }->sendWithAsyncReply(Messages::NetworkStorageManager::Persisted(origin), WTFMove(completionHandler));
}

void WebStorageConnection::persist(const WebCore::ClientOrigin& origin, StorageConnection::PersistCallback&& completionHandler)
{
    Ref { connection() }->sendWithAsyncReply(Messages::NetworkStorageManager::Persist(origin), WTFMove(completionHandler));
}

void WebStorageConnection::getEstimate(WebCore::ClientOrigin&& origin, StorageConnection::GetEstimateCallback&& completionHandler)
{
    Ref { connection() }->sendWithAsyncReply(Messages::NetworkStorageManager::Estimate(origin), [completionHandler = WTFMove(completionHandler)](auto result) mutable {
        if (!result)
            return completionHandler(WebCore::Exception { WebCore::ExceptionCode::TypeError });

        return completionHandler(WTFMove(*result));
    });
}

void WebStorageConnection::fileSystemGetDirectory(WebCore::ClientOrigin&& origin, StorageConnection::GetDirectoryCallback&& completionHandler)
{
    Ref { connection() }->sendWithAsyncReply(Messages::NetworkStorageManager::FileSystemGetDirectory(origin), [completionHandler = WTFMove(completionHandler)](auto result) mutable {
        if (!result)
            return completionHandler(convertToException(result.error()));

        auto identifier = result.value();
        if (!identifier)
            return completionHandler(WebCore::Exception { WebCore::ExceptionCode::UnknownError, "Connection is lost"_s });

        auto connection = RefPtr<WebCore::FileSystemStorageConnection> { &WebProcess::singleton().fileSystemStorageConnection() };
        return completionHandler(std::pair { *identifier, WTFMove(connection) });
    });
}

IPC::Connection& WebStorageConnection::connection()
{
    return WebProcess::singleton().ensureNetworkProcessConnection().connection();
}

} // namespace WebKit
