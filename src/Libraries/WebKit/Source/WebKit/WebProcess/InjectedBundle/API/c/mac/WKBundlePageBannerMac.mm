/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 18, 2024.
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
#import "WKBundlePageBannerMac.h"

#if !PLATFORM(IOS_FAMILY)

#import "APIClient.h"
#import "PageBanner.h"
#import "WKAPICast.h"
#import "WKBundleAPICast.h"
#include <wtf/TZoneMallocInlines.h>

namespace API {
template<> struct ClientTraits<WKBundlePageBannerClientBase> {
    typedef std::tuple<WKBundlePageBannerClientV0> Versions;
};
}

namespace WebKit {
using namespace WebCore;

class PageBannerClientImpl : API::Client<WKBundlePageBannerClientBase>, public PageBanner::Client {
    WTF_MAKE_TZONE_ALLOCATED_INLINE(PageBannerClientImpl);
public:
    explicit PageBannerClientImpl(WKBundlePageBannerClientBase* client)
    {
        initialize(client);
    }

    virtual ~PageBannerClientImpl()
    {
    }

private:
    // PageBanner::Client.
    bool mouseEvent(PageBanner* pageBanner, WebEventType type, WebMouseEventButton button, const IntPoint& position) override
    {
        switch (type) {
        case WebEventType::MouseDown: {
            if (!m_client.mouseDown)
                return false;

            return m_client.mouseDown(toAPI(pageBanner), toAPI(position), toAPI(button), m_client.base.clientInfo);
        }
        case WebEventType::MouseUp: {
            if (!m_client.mouseUp)
                return false;

            return m_client.mouseUp(toAPI(pageBanner), toAPI(position), toAPI(button), m_client.base.clientInfo);
        }
        case WebEventType::MouseMove: {
            if (button == WebMouseEventButton::None) {
                if (!m_client.mouseMoved)
                    return false;

                return m_client.mouseMoved(toAPI(pageBanner), toAPI(position), m_client.base.clientInfo);
            }

            // This is a MouseMove event with a mouse button pressed. Call mouseDragged.
            if (!m_client.mouseDragged)
                return false;

            return m_client.mouseDragged(toAPI(pageBanner), toAPI(position), toAPI(button), m_client.base.clientInfo);
        }

        default:
            return false;
        }
    }
};

}

WKBundlePageBannerRef WKBundlePageBannerCreateBannerWithCALayer(CALayer *layer, int height, WKBundlePageBannerClientBase* wkClient)
{
    if (wkClient && wkClient->version)
        return 0;

    auto clientImpl = makeUnique<WebKit::PageBannerClientImpl>(wkClient);
    return toAPI(&WebKit::PageBanner::create(layer, height, WTFMove(clientImpl)).leakRef());
}

CALayer *WKBundlePageBannerGetLayer(WKBundlePageBannerRef pageBanner)
{
    return WebKit::toImpl(pageBanner)->layer();
}

#endif // !PLATFORM(IOS_FAMILY)
