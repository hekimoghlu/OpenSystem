/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 28, 2025.
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

#include "MediaPlayerEnums.h"
#include <wtf/HashMap.h>
#include <wtf/HashSet.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/Vector.h>
#include <wtf/text/StringHash.h>

namespace WebCore {

class ContentType;

class WEBCORE_EXPORT MIMETypeCache {
    WTF_MAKE_TZONE_ALLOCATED_EXPORT(MIMETypeCache, WEBCORE_EXPORT);
public:
    MIMETypeCache() = default;
    virtual ~MIMETypeCache() = default;

    virtual bool isAvailable() const;
    virtual MediaPlayerEnums::SupportsType canDecodeType(const String&);
    virtual HashSet<String>& supportedTypes();

    bool isEmpty() const;
    bool supportsContainerType(const String&);

protected:
    void addSupportedTypes(const Vector<String>&);

private:
    virtual bool isStaticContainerType(StringView);
    virtual bool isUnsupportedContainerType(const String&);
    virtual void initializeCache(HashSet<String>&);
    virtual bool canDecodeExtendedType(const ContentType&);

    bool shouldOverrideExtendedType(const ContentType&);

    std::optional<HashSet<String>> m_supportedTypes;
    std::optional<UncheckedKeyHashMap<String, MediaPlayerEnums::SupportsType>> m_cachedResults;
};

} // namespace WebCore
