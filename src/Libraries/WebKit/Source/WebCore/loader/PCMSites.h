/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 17, 2022.
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

#include "RegistrableDomain.h"
#include <wtf/text/StringHash.h>

namespace WebCore::PCM {

struct SourceSite {
    explicit SourceSite(const URL& url)
        : registrableDomain { url }
    {
    }

    explicit SourceSite(RegistrableDomain&& domain)
        : registrableDomain { WTFMove(domain) }
    {
    }

    SourceSite isolatedCopy() const & { return SourceSite { registrableDomain.isolatedCopy() }; }
    SourceSite isolatedCopy() && { return SourceSite { WTFMove(registrableDomain).isolatedCopy() }; }

    friend bool operator==(const SourceSite&, const SourceSite&) = default;

    bool matches(const URL& url) const
    {
        return registrableDomain.matches(url);
    }

    RegistrableDomain registrableDomain;
};

struct SourceSiteHash {
    static unsigned hash(const SourceSite& sourceSite)
    {
        return sourceSite.registrableDomain.hash();
    }
    
    static bool equal(const SourceSite& a, const SourceSite& b)
    {
        return a == b;
    }

    static const bool safeToCompareToEmptyOrDeleted = false;
};

struct AttributionDestinationSite {
    AttributionDestinationSite() = default;
    explicit AttributionDestinationSite(const URL& url)
        : registrableDomain { RegistrableDomain { url } }
    {
    }

    explicit AttributionDestinationSite(RegistrableDomain&& domain)
        : registrableDomain { WTFMove(domain) }
    {
    }

    AttributionDestinationSite isolatedCopy() const & { return AttributionDestinationSite { registrableDomain.isolatedCopy() }; }
    AttributionDestinationSite isolatedCopy() && { return AttributionDestinationSite { WTFMove(registrableDomain).isolatedCopy() }; }

    friend bool operator==(const AttributionDestinationSite&, const AttributionDestinationSite&) = default;

    bool matches(const URL& url) const
    {
        return registrableDomain == RegistrableDomain { url };
    }

    RegistrableDomain registrableDomain;
};

struct AttributionDestinationSiteHash {
    static unsigned hash(const AttributionDestinationSite& destinationSite)
    {
        return destinationSite.registrableDomain.hash();
    }
    
    static bool equal(const AttributionDestinationSite& a, const AttributionDestinationSite& b)
    {
        return a == b;
    }

    static const bool safeToCompareToEmptyOrDeleted = false;
};

} // namespace WebCore::PCM

namespace WTF {

template<typename T> struct DefaultHash;

template<> struct DefaultHash<WebCore::PCM::SourceSite> : WebCore::PCM::SourceSiteHash { };
template<> struct HashTraits<WebCore::PCM::SourceSite> : GenericHashTraits<WebCore::PCM::SourceSite> {
    static WebCore::PCM::SourceSite emptyValue() { return WebCore::PCM::SourceSite(WebCore::RegistrableDomain()); }
    static bool isEmptyValue(const WebCore::PCM::SourceSite& value) { return value.registrableDomain.string().isNull(); }
    static void constructDeletedValue(WebCore::PCM::SourceSite& slot) { new (NotNull, &slot.registrableDomain) WebCore::RegistrableDomain(WTF::HashTableDeletedValue); }
    static bool isDeletedValue(const WebCore::PCM::SourceSite& slot) { return slot.registrableDomain.isHashTableDeletedValue(); }
};

template<> struct DefaultHash<WebCore::PCM::AttributionDestinationSite> : WebCore::PCM::AttributionDestinationSiteHash { };
template<> struct HashTraits<WebCore::PCM::AttributionDestinationSite> : GenericHashTraits<WebCore::PCM::AttributionDestinationSite> {
    static WebCore::PCM::AttributionDestinationSite emptyValue() { return { }; }
    static bool isEmptyValue(const WebCore::PCM::AttributionDestinationSite& value) { return value.registrableDomain.string().isNull(); }
    static void constructDeletedValue(WebCore::PCM::AttributionDestinationSite& slot) { new (NotNull, &slot.registrableDomain) WebCore::RegistrableDomain(WTF::HashTableDeletedValue); }
    static bool isDeletedValue(const WebCore::PCM::AttributionDestinationSite& slot) { return slot.registrableDomain.isHashTableDeletedValue(); }
};

} // namespace WTF
