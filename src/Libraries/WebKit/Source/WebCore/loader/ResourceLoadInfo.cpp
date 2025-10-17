/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 4, 2022.
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
#include "ResourceLoadInfo.h"

#if ENABLE(CONTENT_EXTENSIONS)

#include "ContentExtensionActions.h"
#include "RegistrableDomain.h"
#include "SecurityOrigin.h"

namespace WebCore::ContentExtensions {

static_assert(!(ResourceTypeMask & LoadTypeMask), "ResourceTypeMask and LoadTypeMask should be mutually exclusive because they are stored in the same uint32_t");
static_assert(!(ResourceTypeMask & LoadContextMask), "ResourceTypeMask and LoadContextMask should be mutually exclusive because they are stored in the same uint32_t");
static_assert(!(ResourceTypeMask & ActionConditionMask), "ResourceTypeMask and ActionConditionMask should be mutually exclusive because they are stored in the same uint32_t");
static_assert(!(LoadContextMask & LoadTypeMask), "LoadContextMask and LoadTypeMask should be mutually exclusive because they are stored in the same uint32_t");
static_assert(!(LoadContextMask & ActionConditionMask), "LoadContextMask and ActionConditionMask should be mutually exclusive because they are stored in the same uint32_t");
static_assert(!(LoadTypeMask & ActionConditionMask), "LoadTypeMask and ActionConditionMask should be mutually exclusive because they are stored in the same uint32_t");
static_assert(static_cast<uint64_t>(AllResourceFlags) << 32 == ActionFlagMask, "ActionFlagMask should cover all the action flags");

OptionSet<ResourceType> toResourceType(CachedResource::Type type, ResourceRequestRequester requester)
{
    switch (type) {
    case CachedResource::Type::LinkPrefetch:
    case CachedResource::Type::MainResource:
        return { ResourceType::Document };
    case CachedResource::Type::SVGDocumentResource:
        return { ResourceType::SVGDocument };
    case CachedResource::Type::ImageResource:
        return { ResourceType::Image };
    case CachedResource::Type::CSSStyleSheet:
#if ENABLE(XSLT)
    case CachedResource::Type::XSLStyleSheet:
#endif
        return { ResourceType::StyleSheet };

    case CachedResource::Type::Script:
        return { ResourceType::Script };

    case CachedResource::Type::FontResource:
    case CachedResource::Type::SVGFontResource:
        return { ResourceType::Font };

    case CachedResource::Type::MediaResource:
        return { ResourceType::Media };

    case CachedResource::Type::RawResource:
        if (requester == ResourceRequestRequester::XHR
            || requester == ResourceRequestRequester::Fetch)
            return { ResourceType::Fetch };
        FALLTHROUGH;
    case CachedResource::Type::Beacon:
    case CachedResource::Type::Ping:
    case CachedResource::Type::Icon:
#if ENABLE(MODEL_ELEMENT)
    case CachedResource::Type::EnvironmentMapResource:
    case CachedResource::Type::ModelResource:
#endif
#if ENABLE(APPLICATION_MANIFEST)
    case CachedResource::Type::ApplicationManifest:
#endif
        return { ResourceType::Other };

    case CachedResource::Type::TextTrackResource:
        return { ResourceType::Media };

    };
    ASSERT_NOT_REACHED();
    return { };
}

std::optional<OptionSet<ResourceType>> readResourceType(StringView name)
{
    if (name == "document"_s)
        return { ResourceType::Document };
    if (name == "image"_s)
        return { ResourceType::Image };
    if (name == "style-sheet"_s)
        return { ResourceType::StyleSheet };
    if (name == "script"_s)
        return { ResourceType::Script };
    if (name == "font"_s)
        return { ResourceType::Font };
    if (name == "raw"_s)
        return { { ResourceType::Fetch, ResourceType::WebSocket, ResourceType::Other, ResourceType::Ping } };
    if (name == "websocket"_s)
        return { ResourceType::WebSocket };
    if (name == "fetch"_s)
        return { ResourceType::Fetch };
    if (name == "other"_s)
        return { { ResourceType::Other, ResourceType::Ping, ResourceType::CSPReport } };
    if (name == "svg-document"_s)
        return { ResourceType::SVGDocument };
    if (name == "media"_s)
        return { ResourceType::Media };
    if (name == "popup"_s)
        return { ResourceType::Popup };
    if (name == "ping"_s)
        return { ResourceType::Ping };
    if (name == "csp-report"_s)
        return { ResourceType::CSPReport };
    return std::nullopt;
}

std::optional<OptionSet<LoadType>> readLoadType(StringView name)
{
    if (name == "first-party"_s)
        return { LoadType::FirstParty };
    if (name == "third-party"_s)
        return { LoadType::ThirdParty };
    return std::nullopt;
}

std::optional<OptionSet<LoadContext>> readLoadContext(StringView name)
{
    if (name == "top-frame"_s)
        return { LoadContext::TopFrame };
    if (name == "child-frame"_s)
        return { LoadContext::ChildFrame };
    return std::nullopt;
}

bool ResourceLoadInfo::isThirdParty() const
{
    return !RegistrableDomain(mainDocumentURL).matches(resourceURL);
}
    
ResourceFlags ResourceLoadInfo::getResourceFlags() const
{
    ResourceFlags flags = 0;
    ASSERT(!type.isEmpty());
    flags |= type.toRaw();
    flags |= isThirdParty() ? static_cast<ResourceFlags>(LoadType::ThirdParty) : static_cast<ResourceFlags>(LoadType::FirstParty);
    flags |= mainFrameContext ? static_cast<ResourceFlags>(LoadContext::TopFrame) : static_cast<ResourceFlags>(LoadContext::ChildFrame);
    return flags;
}

} // namespace WebCore::ContentExtensions

#endif // ENABLE(CONTENT_EXTENSIONS)
