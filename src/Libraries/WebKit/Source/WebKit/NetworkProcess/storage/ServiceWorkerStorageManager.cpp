/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 27, 2023.
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
#include "ServiceWorkerStorageManager.h"

#include <WebCore/SWRegistrationDatabase.h>
#include <WebCore/ServiceWorkerContextData.h>
#include <WebCore/ServiceWorkerRegistrationKey.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebKit {

WTF_MAKE_TZONE_ALLOCATED_IMPL(ServiceWorkerStorageManager);

ServiceWorkerStorageManager::ServiceWorkerStorageManager(const String& path)
    : m_path(path)
{
}

WebCore::SWRegistrationDatabase* ServiceWorkerStorageManager::ensureDatabase()
{
    if (!m_database && !m_path.isEmpty())
        m_database = makeUnique<WebCore::SWRegistrationDatabase>(m_path);

    return m_database.get();
}

void ServiceWorkerStorageManager::closeFiles()
{
    m_database = nullptr;
}

void ServiceWorkerStorageManager::clearAllRegistrations()
{
    if (auto database = ensureDatabase())
        database->deleteAllFiles();
}

std::optional<Vector<WebCore::ServiceWorkerContextData>> ServiceWorkerStorageManager::importRegistrations()
{
    if (auto database = ensureDatabase())
        return database->importRegistrations();

    return std::nullopt;
}

std::optional<Vector<WebCore::ServiceWorkerScripts>> ServiceWorkerStorageManager::updateRegistrations(const Vector<WebCore::ServiceWorkerContextData>& registrationsToUpdate, const Vector<WebCore::ServiceWorkerRegistrationKey>& registrationsToDelete)
{
    if (auto database = ensureDatabase())
        return database->updateRegistrations(registrationsToUpdate, registrationsToDelete);

    return std::nullopt;
}

} // namespace WebKit
