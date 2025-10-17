/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 25, 2025.
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

#include "FetchOptionsCache.h"
#include "FetchOptionsCredentials.h"
#include "FetchOptionsDestination.h"
#include "FetchOptionsMode.h"
#include "FetchOptionsRedirect.h"
#include "ProcessQualified.h"
#include "ReferrerPolicy.h"
#include "ScriptExecutionContextIdentifier.h"
#include <wtf/Markable.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

struct FetchOptions {
    using Destination = FetchOptionsDestination;
    using Mode = FetchOptionsMode;
    using Credentials = FetchOptionsCredentials;
    using Cache = FetchOptionsCache;
    using Redirect = FetchOptionsRedirect;

    FetchOptions() = default;
    FetchOptions(Destination, Mode, Credentials, Cache, Redirect, ReferrerPolicy, bool, String&&, Markable<WTF::UUID>, Markable<WTF::UUID>);
    FetchOptions isolatedCopy() const & { return { destination, mode, credentials, cache, redirect, referrerPolicy, keepAlive, integrity.isolatedCopy(), clientIdentifier, resultingClientIdentifier }; }
    FetchOptions isolatedCopy() && { return { destination, mode, credentials, cache, redirect, referrerPolicy, keepAlive, WTFMove(integrity).isolatedCopy(), clientIdentifier, resultingClientIdentifier }; }

    template<class Encoder> void encodePersistent(Encoder&) const;
    template<class Decoder> static WARN_UNUSED_RETURN bool decodePersistent(Decoder&, FetchOptions&);

    Destination destination { Destination::EmptyString };
    Mode mode { Mode::NoCors };
    Credentials credentials { Credentials::Omit };
    Cache cache { Cache::Default };
    Redirect redirect { Redirect::Follow };
    ReferrerPolicy referrerPolicy { ReferrerPolicy::EmptyString };
    bool keepAlive { false };
    String integrity;
    Markable<WTF::UUID> clientIdentifier; // Identifier of https://fetch.spec.whatwg.org/#concept-request-client
    Markable<WTF::UUID> resultingClientIdentifier; // Identifier of https://fetch.spec.whatwg.org/#concept-request-reserved-client
};

inline FetchOptions::FetchOptions(Destination destination, Mode mode, Credentials credentials, Cache cache, Redirect redirect, ReferrerPolicy referrerPolicy, bool keepAlive, String&& integrity, Markable<WTF::UUID> clientIdentifier, Markable<WTF::UUID> resultingClientIdentifier)
    : destination(destination)
    , mode(mode)
    , credentials(credentials)
    , cache(cache)
    , redirect(redirect)
    , referrerPolicy(referrerPolicy)
    , keepAlive(keepAlive)
    , integrity(WTFMove(integrity))
    , clientIdentifier(clientIdentifier)
    , resultingClientIdentifier(resultingClientIdentifier)
{
}

inline bool isPotentialNavigationOrSubresourceRequest(FetchOptions::Destination destination)
{
    return destination == FetchOptions::Destination::Object
        || destination == FetchOptions::Destination::Embed;
}

// https://fetch.spec.whatwg.org/#navigation-request
inline bool isNavigationRequest(FetchOptions::Destination destination)
{
    return destination == FetchOptions::Destination::Document
        || destination == FetchOptions::Destination::Iframe
        || destination == FetchOptions::Destination::Object
        || destination == FetchOptions::Destination::Embed;
}

// https://fetch.spec.whatwg.org/#non-subresource-request
inline bool isNonSubresourceRequest(FetchOptions::Destination destination)
{
    return destination == FetchOptions::Destination::Document
        || destination == FetchOptions::Destination::Iframe
        || destination == FetchOptions::Destination::Report
        || destination == FetchOptions::Destination::Serviceworker
        || destination == FetchOptions::Destination::Sharedworker
        || destination == FetchOptions::Destination::Worker;
}

inline bool isScriptLikeDestination(FetchOptions::Destination destination)
{
    return destination == FetchOptions::Destination::Audioworklet
        || destination == FetchOptions::Destination::Paintworklet
        || destination == FetchOptions::Destination::Script
        || destination == FetchOptions::Destination::Serviceworker
        || destination == FetchOptions::Destination::Sharedworker
        || destination == FetchOptions::Destination::Worker;
}

}

namespace WTF {

template<> struct EnumTraitsForPersistence<WebCore::FetchOptions::Destination> {
    using values = EnumValues<
        WebCore::FetchOptions::Destination,
        WebCore::FetchOptions::Destination::EmptyString,
        WebCore::FetchOptions::Destination::Audio,
        WebCore::FetchOptions::Destination::Audioworklet,
        WebCore::FetchOptions::Destination::Document,
        WebCore::FetchOptions::Destination::Embed,
        WebCore::FetchOptions::Destination::Environmentmap,
        WebCore::FetchOptions::Destination::Font,
        WebCore::FetchOptions::Destination::Image,
        WebCore::FetchOptions::Destination::Iframe,
        WebCore::FetchOptions::Destination::Manifest,
        WebCore::FetchOptions::Destination::Model,
        WebCore::FetchOptions::Destination::Object,
        WebCore::FetchOptions::Destination::Paintworklet,
        WebCore::FetchOptions::Destination::Report,
        WebCore::FetchOptions::Destination::Script,
        WebCore::FetchOptions::Destination::Serviceworker,
        WebCore::FetchOptions::Destination::Sharedworker,
        WebCore::FetchOptions::Destination::Style,
        WebCore::FetchOptions::Destination::Track,
        WebCore::FetchOptions::Destination::Video,
        WebCore::FetchOptions::Destination::Worker,
        WebCore::FetchOptions::Destination::Xslt
    >;
};

template<> struct EnumTraitsForPersistence<WebCore::FetchOptions::Mode> {
    using values = EnumValues<
        WebCore::FetchOptions::Mode,
        WebCore::FetchOptions::Mode::Navigate,
        WebCore::FetchOptions::Mode::SameOrigin,
        WebCore::FetchOptions::Mode::NoCors,
        WebCore::FetchOptions::Mode::Cors
    >;
};


template<> struct EnumTraitsForPersistence<WebCore::FetchOptions::Credentials> {
    using values = EnumValues<
        WebCore::FetchOptions::Credentials,
        WebCore::FetchOptions::Credentials::Omit,
        WebCore::FetchOptions::Credentials::SameOrigin,
        WebCore::FetchOptions::Credentials::Include
    >;
};

template<> struct EnumTraitsForPersistence<WebCore::FetchOptions::Cache> {
    using values = EnumValues<
        WebCore::FetchOptions::Cache,
        WebCore::FetchOptions::Cache::Default,
        WebCore::FetchOptions::Cache::NoStore,
        WebCore::FetchOptions::Cache::Reload,
        WebCore::FetchOptions::Cache::NoCache,
        WebCore::FetchOptions::Cache::ForceCache,
        WebCore::FetchOptions::Cache::OnlyIfCached
    >;
};

template<> struct EnumTraitsForPersistence<WebCore::FetchOptions::Redirect> {
    using values = EnumValues<
        WebCore::FetchOptions::Redirect,
        WebCore::FetchOptions::Redirect::Follow,
        WebCore::FetchOptions::Redirect::Error,
        WebCore::FetchOptions::Redirect::Manual
    >;
};

}

namespace WebCore {

template<class Encoder>
inline void FetchOptions::encodePersistent(Encoder& encoder) const
{
    // Changes to encoding here should bump NetworkCache Storage format version.
    encoder << destination;
    encoder << mode;
    encoder << credentials;
    encoder << cache;
    encoder << redirect;
    encoder << referrerPolicy;
    encoder << integrity;
    encoder << keepAlive;
}

template<class Decoder>
inline bool FetchOptions::decodePersistent(Decoder& decoder, FetchOptions& options)
{
    std::optional<FetchOptions::Destination> destination;
    decoder >> destination;
    if (!destination)
        return false;

    std::optional<FetchOptions::Mode> mode;
    decoder >> mode;
    if (!mode)
        return false;

    std::optional<FetchOptions::Credentials> credentials;
    decoder >> credentials;
    if (!credentials)
        return false;

    std::optional<FetchOptions::Cache> cache;
    decoder >> cache;
    if (!cache)
        return false;

    std::optional<FetchOptions::Redirect> redirect;
    decoder >> redirect;
    if (!redirect)
        return false;

    std::optional<ReferrerPolicy> referrerPolicy;
    decoder >> referrerPolicy;
    if (!referrerPolicy)
        return false;

    std::optional<String> integrity;
    decoder >> integrity;
    if (!integrity)
        return false;

    std::optional<bool> keepAlive;
    decoder >> keepAlive;
    if (!keepAlive)
        return false;

    options.destination = *destination;
    options.mode = *mode;
    options.credentials = *credentials;
    options.cache = *cache;
    options.redirect = *redirect;
    options.referrerPolicy = *referrerPolicy;
    options.integrity = WTFMove(*integrity);
    options.keepAlive = *keepAlive;

    return true;
}

} // namespace WebCore
