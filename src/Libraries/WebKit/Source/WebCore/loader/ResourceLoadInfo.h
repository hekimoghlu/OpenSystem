/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 4, 2022.
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

#if ENABLE(CONTENT_EXTENSIONS)

#include "CachedResource.h"
#include <wtf/OptionSet.h>
#include <wtf/URL.h>

namespace WebCore::ContentExtensions {

enum class ActionCondition : uint32_t {
    None = 0x00000,
    IfTopURL = 0x20000,
    UnlessTopURL = 0x40000,
    IfFrameURL = 0x60000,
    UnlessFrameURL = 0x80000
};
static constexpr uint32_t ActionConditionMask = 0xE0000;

enum class ResourceType : uint32_t {
    Document = 0x0001,
    Image = 0x0002,
    StyleSheet = 0x0004,
    Script = 0x0008,
    Font = 0x0010,
    SVGDocument = 0x0020,
    Media = 0x0040,
    Popup = 0x0080,
    Ping = 0x0100,
    Fetch = 0x0200,
    WebSocket = 0x0400,
    Other = 0x0800,
    CSPReport = 0x10000,
};
static constexpr uint32_t ResourceTypeMask = 0x10FFF;

enum class LoadType : uint32_t {
    FirstParty = 0x1000,
    ThirdParty = 0x2000,
};
static constexpr uint32_t LoadTypeMask = 0x3000;

enum class LoadContext : uint32_t {
    TopFrame = 0x4000,
    ChildFrame = 0x8000,
};
static constexpr uint32_t LoadContextMask = 0xC000;

using ResourceFlags = uint32_t;

constexpr ResourceFlags AllResourceFlags = LoadTypeMask | ResourceTypeMask | LoadContextMask | ActionConditionMask;

// The first 32 bits of a uint64_t action are used for the action location.
// The next 20 bits are used for the flags (ResourceType, LoadType, LoadContext, ActionCondition).
// The values -1 and -2 are used for removed and empty values in HashTables.
static constexpr uint64_t ActionFlagMask = 0x000FFFFF00000000;

OptionSet<ResourceType> toResourceType(CachedResource::Type, ResourceRequestRequester);
std::optional<OptionSet<ResourceType>> readResourceType(StringView);
std::optional<OptionSet<LoadType>> readLoadType(StringView);
std::optional<OptionSet<LoadContext>> readLoadContext(StringView);

struct ResourceLoadInfo {
    URL resourceURL;
    URL mainDocumentURL;
    URL frameURL;
    OptionSet<ResourceType> type;
    bool mainFrameContext { false };

    bool isThirdParty() const;
    ResourceFlags getResourceFlags() const;
    ResourceLoadInfo isolatedCopy() const & { return { resourceURL.isolatedCopy(), mainDocumentURL.isolatedCopy(), frameURL.isolatedCopy(), type, mainFrameContext }; }
    ResourceLoadInfo isolatedCopy() && { return { WTFMove(resourceURL).isolatedCopy(), WTFMove(mainDocumentURL).isolatedCopy(), WTFMove(frameURL).isolatedCopy(), type, mainFrameContext }; }
};

} // namespace WebCore::ContentExtensions

#endif
