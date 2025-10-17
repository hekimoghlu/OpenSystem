/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 1, 2023.
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
#include "IDBDatabaseIdentifier.h"

#include "SecurityOrigin.h"
#include <wtf/FileSystem.h>
#include <wtf/Ref.h>
#include <wtf/text/MakeString.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

IDBDatabaseIdentifier::IDBDatabaseIdentifier(const String& databaseName, SecurityOriginData&& openingOrigin, SecurityOriginData&& mainFrameOrigin, bool isTransient)
    : m_databaseName(databaseName)
    , m_origin { WTFMove(mainFrameOrigin), WTFMove(openingOrigin) }
    , m_isTransient(isTransient)
{
    // The empty string is a valid database name, but a null string is not.
    ASSERT(!databaseName.isNull());
}

IDBDatabaseIdentifier IDBDatabaseIdentifier::isolatedCopy() const &
{
    IDBDatabaseIdentifier identifier;
    identifier.m_databaseName = m_databaseName.isolatedCopy();
    identifier.m_origin = m_origin.isolatedCopy();
    identifier.m_isTransient = m_isTransient;
    return identifier;
}

IDBDatabaseIdentifier IDBDatabaseIdentifier::isolatedCopy() &&
{
    IDBDatabaseIdentifier identifier;
    identifier.m_databaseName = WTFMove(m_databaseName).isolatedCopy();
    identifier.m_origin = WTFMove(m_origin).isolatedCopy();
    identifier.m_isTransient = m_isTransient;
    return identifier;
}

String IDBDatabaseIdentifier::databaseDirectoryRelativeToRoot(const String& rootDirectory, ASCIILiteral versionString) const
{
    return databaseDirectoryRelativeToRoot(m_origin, rootDirectory, versionString);
}

String IDBDatabaseIdentifier::databaseDirectoryRelativeToRoot(const ClientOrigin& origin, const String& rootDirectory, ASCIILiteral versionString)
{
    String versionDirectory = FileSystem::pathByAppendingComponent(rootDirectory, StringView { versionString });
    String mainFrameDirectory = FileSystem::pathByAppendingComponent(versionDirectory, origin.topOrigin.databaseIdentifier());

    // If the opening origin and main frame origins are the same, there is no partitioning.
    if (origin.topOrigin == origin.clientOrigin)
        return mainFrameDirectory;

    return FileSystem::pathByAppendingComponent(mainFrameDirectory, origin.clientOrigin.databaseIdentifier());
}

String IDBDatabaseIdentifier::optionalDatabaseDirectoryRelativeToRoot(const ClientOrigin& origin, const String& rootDirectory, ASCIILiteral versionString)
{
    auto topOriginURL = origin.topOrigin.toURL();
    auto clientOriginURL = origin.clientOrigin.toURL();
    if (!topOriginURL.isValid() || !clientOriginURL.isValid())
        return { };

    return databaseDirectoryRelativeToRoot(origin, rootDirectory, versionString);
}

#if !LOG_DISABLED
String IDBDatabaseIdentifier::loggingString() const
{
    return makeString(m_databaseName, '@', m_origin.topOrigin.debugString(), ':', m_origin.clientOrigin.debugString(), m_isTransient ? ", transient"_s : ""_s);
}
#endif

} // namespace WebCore
