/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 27, 2024.
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
#include "ArchiveFactory.h"

#include "MIMETypeRegistry.h"

#if ENABLE(WEB_ARCHIVE) && USE(CF)
#include "LegacyWebArchive.h"
#endif
#if ENABLE(MHTML)
#include "MHTMLArchive.h"
#endif

#include <wtf/HashMap.h>
#include <wtf/HashSet.h>
#include <wtf/NeverDestroyed.h>
#include <wtf/StdLibExtras.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

typedef RefPtr<Archive> RawDataCreationFunction(const URL&, FragmentedSharedBuffer&);
typedef HashMap<String, RawDataCreationFunction*, ASCIICaseInsensitiveHash> ArchiveMIMETypesMap;

// The create functions in the archive classes return RefPtr to concrete subclasses
// of Archive. This adaptor makes the functions have a uniform return type.
template<typename ArchiveClass> static RefPtr<Archive> archiveFactoryCreate(const URL& url, FragmentedSharedBuffer& buffer)
{
    return ArchiveClass::create(url, buffer);
}

static ArchiveMIMETypesMap createArchiveMIMETypesMap()
{
    ArchiveMIMETypesMap map;

#if ENABLE(WEB_ARCHIVE) && USE(CF)
    map.add("application/x-webarchive"_s, archiveFactoryCreate<LegacyWebArchive>);
#endif

#if ENABLE(MHTML)
    map.add("multipart/related"_s, archiveFactoryCreate<MHTMLArchive>);
    map.add("application/x-mimearchive"_s, archiveFactoryCreate<MHTMLArchive>);
#endif

    return map;
}

static ArchiveMIMETypesMap& archiveMIMETypes()
{
    static NeverDestroyed<ArchiveMIMETypesMap> map = createArchiveMIMETypesMap();
    return map;
}

bool ArchiveFactory::isArchiveMIMEType(const String& mimeType)
{
    return !mimeType.isEmpty() && archiveMIMETypes().contains(mimeType);
}

RefPtr<Archive> ArchiveFactory::create(const URL& url, FragmentedSharedBuffer* data, const String& mimeType)
{
    if (!data)
        return nullptr;
    if (mimeType.isEmpty())
        return nullptr;
    auto* function = archiveMIMETypes().get(mimeType);
    if (!function)
        return nullptr;
    return function(url, *data);
}

void ArchiveFactory::registerKnownArchiveMIMETypes(HashSet<String, ASCIICaseInsensitiveHash>& mimeTypes)
{
    for (auto& mimeType : archiveMIMETypes().keys())
        mimeTypes.add(mimeType);
}

}
