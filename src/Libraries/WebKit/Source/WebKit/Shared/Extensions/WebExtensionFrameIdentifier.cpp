/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 10, 2022.
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
#include "WebExtensionFrameIdentifier.h"

#include "FrameInfoData.h"
#include "WebFrame.h"
#include "WebPage.h"
#include <WebCore/Frame.h>

namespace WebKit {

WebCore::FrameIdentifier toWebCoreFrameIdentifier(const WebExtensionFrameIdentifier& identifier, const WebPage& page)
{
    if (isMainFrame(identifier))
        return page.mainWebFrame().frameID();

    return { ObjectIdentifier<WebCore::FrameIdentifierType> { identifier.toUInt64() }, WebCore::Process::identifier() };
}

bool matchesFrame(const WebExtensionFrameIdentifier& identifier, const WebFrame& frame)
{
    if (RefPtr coreFrame = frame.coreFrame(); coreFrame && coreFrame->isMainFrame() && isMainFrame(identifier))
        return true;

    if (RefPtr page = frame.page(); page && &page->mainWebFrame() == &frame && isMainFrame(identifier))
        return true;

    return frame.frameID().object().toUInt64() == identifier.toUInt64() && !frame.isMainFrame();
}

WebExtensionFrameIdentifier toWebExtensionFrameIdentifier(std::optional<WebCore::FrameIdentifier> frameIdentifier)
{
    if (!frameIdentifier) {
        ASSERT_NOT_REACHED();
        return WebExtensionFrameConstants::NoneIdentifier;
    }

    auto identifierAsUInt64 = frameIdentifier->object().toUInt64();
    if (!WebExtensionFrameIdentifier::isValidIdentifier(identifierAsUInt64)) {
        ASSERT_NOT_REACHED();
        return WebExtensionFrameConstants::NoneIdentifier;
    }

    return WebExtensionFrameIdentifier { identifierAsUInt64 };
}

WebExtensionFrameIdentifier toWebExtensionFrameIdentifier(const WebFrame& frame)
{
    if (RefPtr coreFrame = frame.coreFrame(); coreFrame && coreFrame->isMainFrame())
        return WebExtensionFrameConstants::MainFrameIdentifier;

    if (RefPtr page = frame.page(); page && &page->mainWebFrame() == &frame)
        return WebExtensionFrameConstants::MainFrameIdentifier;

    return toWebExtensionFrameIdentifier(std::optional { frame.frameID() });
}

WebExtensionFrameIdentifier toWebExtensionFrameIdentifier(const FrameInfoData& frameInfoData)
{
    if (frameInfoData.isMainFrame)
        return WebExtensionFrameConstants::MainFrameIdentifier;

    return toWebExtensionFrameIdentifier(frameInfoData.frameID);
}

std::optional<WebExtensionFrameIdentifier> toWebExtensionFrameIdentifier(double identifier)
{
    if (identifier == WebExtensionFrameConstants::MainFrame)
        return WebExtensionFrameConstants::MainFrameIdentifier;

    if (identifier == WebExtensionFrameConstants::None)
        return WebExtensionFrameConstants::NoneIdentifier;

    if (!std::isfinite(identifier) || identifier <= 0 || identifier >= static_cast<double>(WebExtensionFrameConstants::NoneIdentifier.toUInt64()))
        return std::nullopt;

    double integral;
    if (std::modf(identifier, &integral) != 0.0) {
        // Only integral numbers can be used.
        return std::nullopt;
    }

    auto identifierAsUInt64 = static_cast<uint64_t>(identifier);
    if (!WebExtensionFrameIdentifier::isValidIdentifier(identifierAsUInt64)) {
        ASSERT_NOT_REACHED();
        return std::nullopt;
    }

    return WebExtensionFrameIdentifier { identifierAsUInt64 };
}

}
