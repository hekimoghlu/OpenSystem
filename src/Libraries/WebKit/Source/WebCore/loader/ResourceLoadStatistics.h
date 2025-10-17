/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 13, 2023.
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

#include "CanvasActivityRecord.h"
#include "RegistrableDomain.h"
#include <wtf/ArgumentCoder.h>
#include <wtf/HashCountedSet.h>
#include <wtf/HashSet.h>
#include <wtf/OptionSet.h>
#include <wtf/URL.h>
#include <wtf/WallTime.h>
#include <wtf/text/StringHash.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

class KeyedDecoder;
class KeyedEncoder;

enum class NavigatorAPIsAccessed : uint64_t {
    AppVersion = 1 << 0,
    UserAgent = 1 << 1,
    Plugins = 1 << 2,
    MimeTypes = 1 << 3,
    CookieEnabled = 1 << 4,
};

enum class ScreenAPIsAccessed : uint64_t {
    Height = 1 << 0,
    Width = 1 << 1,
    ColorDepth = 1 << 2,
    AvailLeft = 1 << 3,
    AvailTop = 1 << 4,
    AvailHeight = 1 << 5,
    AvailWidth = 1 << 6,
};

struct ResourceLoadStatistics {
    WTF_MAKE_STRUCT_FAST_ALLOCATED;
    explicit ResourceLoadStatistics(const RegistrableDomain& domain)
        : registrableDomain { domain }
    {
    }

    ResourceLoadStatistics() = default;

    ResourceLoadStatistics(const ResourceLoadStatistics&) = delete;
    ResourceLoadStatistics& operator=(const ResourceLoadStatistics&) = delete;
    ResourceLoadStatistics(ResourceLoadStatistics&&) = default;
    ResourceLoadStatistics& operator=(ResourceLoadStatistics&&) = default;

    static constexpr Seconds NoExistingTimestamp { -1 };

    WEBCORE_EXPORT static WallTime reduceTimeResolution(WallTime);

    WEBCORE_EXPORT void encode(KeyedEncoder&) const;
    WEBCORE_EXPORT bool decode(KeyedDecoder&, unsigned modelVersion);

    WEBCORE_EXPORT String toString() const;

    WEBCORE_EXPORT void merge(const ResourceLoadStatistics&);

    RegistrableDomain registrableDomain;

    WallTime lastSeen;

    // User interaction
    bool hadUserInteraction { false };
    // Timestamp. Default value is negative, 0 means it was reset.
    WallTime mostRecentUserInteractionTime { WallTime::fromRawSeconds(-1) };
    bool grandfathered { false };

    // Storage access
    HashSet<RegistrableDomain> storageAccessUnderTopFrameDomains;

    // Top frame stats
    HashSet<RegistrableDomain> topFrameUniqueRedirectsTo;
    HashSet<RegistrableDomain> topFrameUniqueRedirectsToSinceSameSiteStrictEnforcement;
    HashSet<RegistrableDomain> topFrameUniqueRedirectsFrom;
    HashSet<RegistrableDomain> topFrameLinkDecorationsFrom;
    bool gotLinkDecorationFromPrevalentResource { false };
    HashSet<RegistrableDomain> topFrameLoadedThirdPartyScripts;

    // Subframe stats
    HashSet<RegistrableDomain> subframeUnderTopFrameDomains;

    // Subresource stats
    HashSet<RegistrableDomain> subresourceUnderTopFrameDomains;
    HashSet<RegistrableDomain> subresourceUniqueRedirectsTo;
    HashSet<RegistrableDomain> subresourceUniqueRedirectsFrom;

    // Prevalent resource stats
    bool isPrevalentResource { false };
    bool isVeryPrevalentResource { false };
    unsigned dataRecordsRemoved { 0 };
    unsigned timesAccessedAsFirstPartyDueToUserInteraction { 0 };
    unsigned timesAccessedAsFirstPartyDueToStorageAccessAPI { 0 };

    enum class IsEphemeral : bool { No, Yes };

#if ENABLE(WEB_API_STATISTICS)
    // This set represents the registrable domain of the top frame where web API
    // were used in the top frame or one of its subframes.
    HashSet<RegistrableDomain> topFrameRegistrableDomainsWhichAccessedWebAPIs;
    HashSet<String> fontsFailedToLoad;
    HashSet<String> fontsSuccessfullyLoaded;
    CanvasActivityRecord canvasActivityRecord;
    OptionSet<NavigatorAPIsAccessed> navigatorFunctionsAccessed;
    OptionSet<ScreenAPIsAccessed> screenFunctionsAccessed;
#endif
    friend struct IPC::ArgumentCoder<ResourceLoadStatistics, void>;
};

} // namespace WebCore
