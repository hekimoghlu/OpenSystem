/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 4, 2023.
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
#include "InspectorDatabaseResource.h"

#include "Database.h"


namespace WebCore {

using namespace Inspector;

static int nextUnusedId = 1;

Ref<InspectorDatabaseResource> InspectorDatabaseResource::create(Database& database, const String& domain, const String& name, const String& version)
{
    return adoptRef(*new InspectorDatabaseResource(database, domain, name, version));
}

InspectorDatabaseResource::InspectorDatabaseResource(Database& database, const String& domain, const String& name, const String& version)
    : m_database(database)
    , m_id(String::number(nextUnusedId++))
    , m_domain(domain)
    , m_name(name)
    , m_version(version)
{
}

void InspectorDatabaseResource::bind(Inspector::DatabaseFrontendDispatcher& databaseFrontendDispatcher)
{
    auto jsonObject = Inspector::Protocol::Database::Database::create()
        .setId(m_id)
        .setDomain(m_domain)
        .setName(m_name)
        .setVersion(m_version)
        .release();
    databaseFrontendDispatcher.addDatabase(WTFMove(jsonObject));
}

} // namespace WebCore
