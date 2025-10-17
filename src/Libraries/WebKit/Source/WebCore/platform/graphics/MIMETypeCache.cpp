/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 14, 2021.
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
#include "MIMETypeCache.h"

#include "ContentType.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(MIMETypeCache);

HashSet<String>& MIMETypeCache::supportedTypes()
{
    if (!m_supportedTypes) {
        m_supportedTypes = HashSet<String> { };
        initializeCache(*m_supportedTypes);
    }

    return *m_supportedTypes;
}

bool MIMETypeCache::supportsContainerType(const String& containerType)
{
    if (!isAvailable() || containerType.isEmpty())
        return false;

    if (isUnsupportedContainerType(containerType))
        return false;

    return isStaticContainerType(containerType) || supportedTypes().contains(containerType);
}

MediaPlayerEnums::SupportsType MIMETypeCache::canDecodeType(const String& mimeType)
{
    if (mimeType.isEmpty())
        return MediaPlayerEnums::SupportsType::IsNotSupported;

    if (m_cachedResults) {
        auto it = m_cachedResults->find(mimeType);
        if (it != m_cachedResults->end())
            return it->value;
    }

    auto result = MediaPlayerEnums::SupportsType::IsNotSupported;
    do {
        if (!isAvailable() || mimeType.isEmpty())
            break;

        auto contentType = ContentType { mimeType };
        auto containerType = contentType.containerType();
        if (!supportsContainerType(containerType))
            break;

        if (contentType.codecs().isEmpty()) {
            result = MediaPlayerEnums::SupportsType::MayBeSupported;
            break;
        }

        if (shouldOverrideExtendedType(contentType)) {
            result = MediaPlayerEnums::SupportsType::IsSupported;
            break;
        }

        if (canDecodeExtendedType(contentType))
            result = MediaPlayerEnums::SupportsType::IsSupported;

    } while (0);

    if (!m_cachedResults)
        m_cachedResults = UncheckedKeyHashMap<String, MediaPlayerEnums::SupportsType>();
    m_cachedResults->add(mimeType, result);

    return result;
}

void MIMETypeCache::addSupportedTypes(const Vector<String>& newTypes)
{
    if (!m_supportedTypes)
        m_supportedTypes = HashSet<String> { };

    for (auto& type : newTypes)
        m_supportedTypes->add(type);
}

bool MIMETypeCache::isStaticContainerType(StringView)
{
    return false;
}

bool MIMETypeCache::isUnsupportedContainerType(const String&)
{
    return false;
}

bool MIMETypeCache::isAvailable() const
{
    return true;
}

bool MIMETypeCache::isEmpty() const
{
    return m_supportedTypes && m_supportedTypes->isEmpty();
}

void MIMETypeCache::initializeCache(HashSet<String>&)
{
}

bool MIMETypeCache::canDecodeExtendedType(const ContentType&)
{
    return false;
}

bool MIMETypeCache::shouldOverrideExtendedType(const ContentType& type)
{
    ASSERT(canDecodeType(type.containerType()) != MediaPlayerEnums::SupportsType::IsNotSupported);

    // Some sites (e.g. Modernizr) use 'audio/mpeg; codecs="mp3"' even though
    // it is not RFC 3003 compliant.
    if (equalLettersIgnoringASCIICase(type.containerType(), "audio/mpeg"_s)) {
        auto codecs = type.codecs();
        return codecs.size() == 1 && codecs[0] == "mp3"_s;
    }

    return false;
}

}
