/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 4, 2024.
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

#include <WebCore/FrameIdentifier.h>
#include <wtf/ObjectIdentifier.h>

OBJC_CLASS WKFrameInfo;

namespace WebKit {

class WebFrame;
class WebPage;
struct FrameInfoData;

struct WebExtensionFrameIdentifierType;
using WebExtensionFrameIdentifier = ObjectIdentifier<WebExtensionFrameIdentifierType>;

namespace WebExtensionFrameConstants {

static constexpr double MainFrame { 0 };
static constexpr double None { -1 };

static constexpr const WebExtensionFrameIdentifier MainFrameIdentifier { std::numeric_limits<uint64_t>::max() - 1 };
static constexpr const WebExtensionFrameIdentifier NoneIdentifier { std::numeric_limits<uint64_t>::max() - 2 };

}

inline bool isMainFrame(WebExtensionFrameIdentifier identifier)
{
    return identifier == WebExtensionFrameConstants::MainFrameIdentifier;
}

inline bool isMainFrame(std::optional<WebExtensionFrameIdentifier> identifier)
{
    return identifier && isMainFrame(identifier.value());
}

inline bool isNone(WebExtensionFrameIdentifier identifier)
{
    return identifier == WebExtensionFrameConstants::NoneIdentifier;
}

inline bool isNone(std::optional<WebExtensionFrameIdentifier> identifier)
{
    return identifier && isNone(identifier.value());
}

inline bool isValid(std::optional<WebExtensionFrameIdentifier> identifier)
{
    return identifier && !isNone(identifier.value());
}

WebCore::FrameIdentifier toWebCoreFrameIdentifier(const WebExtensionFrameIdentifier&, const WebPage&);

bool matchesFrame(const WebExtensionFrameIdentifier&, const WebFrame&);

WebExtensionFrameIdentifier toWebExtensionFrameIdentifier(std::optional<WebCore::FrameIdentifier>);
WebExtensionFrameIdentifier toWebExtensionFrameIdentifier(const WebFrame&);
WebExtensionFrameIdentifier toWebExtensionFrameIdentifier(const FrameInfoData&);

#ifdef __OBJC__
WebExtensionFrameIdentifier toWebExtensionFrameIdentifier(WKFrameInfo *);
#endif

std::optional<WebExtensionFrameIdentifier> toWebExtensionFrameIdentifier(double identifier);

inline double toWebAPI(const WebExtensionFrameIdentifier& identifier)
{
    if (isMainFrame(identifier))
        return WebExtensionFrameConstants::MainFrame;

    if (isNone(identifier))
        return WebExtensionFrameConstants::None;

    return static_cast<double>(identifier.toUInt64());
}

}
