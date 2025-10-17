/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 28, 2021.
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
#import "config.h"
#import "MediaCapability.h"

#if ENABLE(EXTENSION_CAPABILITIES)

#import <BrowserEngineKit/BECapability.h>
#import <WebCore/SecurityOrigin.h>
#import <wtf/text/WTFString.h>

namespace WebKit {

static RetainPtr<BEMediaEnvironment> createMediaEnvironment(const URL& webPageURL)
{
    NSURL *protocolHostAndPortURL = URL { webPageURL.protocolHostAndPort() };
    RELEASE_ASSERT(protocolHostAndPortURL);
    return adoptNS([[BEMediaEnvironment alloc] initWithWebPageURL:protocolHostAndPortURL]);
}

Ref<MediaCapability> MediaCapability::create(URL&& url)
{
    return adoptRef(*new MediaCapability(WTFMove(url)));
}

MediaCapability::MediaCapability(URL&& webPageURL)
    : m_webPageURL { WTFMove(webPageURL) }
    , m_mediaEnvironment { createMediaEnvironment(m_webPageURL) }
{
    setPlatformCapability([BEProcessCapability mediaPlaybackAndCaptureWithEnvironment:m_mediaEnvironment.get()]);
}

bool MediaCapability::isActivatingOrActive() const
{
    switch (m_state) {
    case State::Inactive:
    case State::Deactivating:
        return false;
    case State::Activating:
    case State::Active:
        return true;
    }

    RELEASE_ASSERT_NOT_REACHED();
    return false;
}

String MediaCapability::environmentIdentifier() const
{
#if USE(EXTENSIONKIT)
    xpc_object_t xpcObject = [m_mediaEnvironment createXPCRepresentation];
    if (!xpcObject)
        return emptyString();
    return xpc_dictionary_get_wtfstring(xpcObject, "identifier"_s);
#endif

    return { };
}

} // namespace WebKit

#endif // ENABLE(EXTENSION_CAPABILITIES)
