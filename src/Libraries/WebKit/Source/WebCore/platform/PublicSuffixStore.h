/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 1, 2023.
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

#include "PublicSuffix.h"
#include <wtf/HashMap.h>
#include <wtf/HashSet.h>
#include <wtf/text/StringHash.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

class PublicSuffixStore {
public:
    WEBCORE_EXPORT static PublicSuffixStore& singleton();

    // https://url.spec.whatwg.org/#host-public-suffix
    WEBCORE_EXPORT bool isPublicSuffix(StringView domain) const;
    WEBCORE_EXPORT PublicSuffix publicSuffix(const URL&) const;
    WEBCORE_EXPORT String topPrivatelyControlledDomain(StringView host) const;
    WEBCORE_EXPORT void clearHostTopPrivatelyControlledDomainCache();
    WEBCORE_EXPORT String topPrivatelyControlledDomainWithoutPublicSuffix(StringView host) const;
    WEBCORE_EXPORT String domainWithoutPublicSuffix(StringView domain) const;

#if PLATFORM(COCOA)
    WEBCORE_EXPORT void enablePublicSuffixCache();
    WEBCORE_EXPORT void addPublicSuffix(const PublicSuffix&);
#endif

private:
    friend LazyNeverDestroyed<PublicSuffixStore>;
    PublicSuffixStore() = default;

    bool platformIsPublicSuffix(StringView domain) const;
    String platformTopPrivatelyControlledDomain(StringView host) const;

    mutable Lock m_HostTopPrivatelyControlledDomainCacheLock;
    mutable UncheckedKeyHashMap<String, String, ASCIICaseInsensitiveHash> m_hostTopPrivatelyControlledDomainCache WTF_GUARDED_BY_LOCK(m_HostTopPrivatelyControlledDomainCacheLock);
#if PLATFORM(COCOA)
    mutable Lock m_publicSuffixCacheLock;
    std::optional<UncheckedKeyHashSet<PublicSuffix>> m_publicSuffixCache WTF_GUARDED_BY_LOCK(m_publicSuffixCacheLock);
#endif
};

} // namespace WebCore
