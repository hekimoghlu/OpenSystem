/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 13, 2022.
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
#pragma once

#include <JavaScriptCore/InspectorFrontendDispatchers.h>
#include <wtf/RefCounted.h>
#include <wtf/RefPtr.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

class Database;

class InspectorDatabaseResource : public RefCounted<InspectorDatabaseResource> {
public:
    static Ref<InspectorDatabaseResource> create(Database&, const String& domain, const String& name, const String& version);

    void bind(Inspector::DatabaseFrontendDispatcher&);

    Database& database() const { return m_database.get(); }
    void setDatabase(Database& database) { m_database = database; }

    String id() const { return m_id; }

private:
    InspectorDatabaseResource(Database&, const String& domain, const String& name, const String& version);

    Ref<Database> m_database;
    String m_id;
    String m_domain;
    String m_name;
    String m_version;
};

} // namespace WebCore
