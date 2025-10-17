/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 13, 2023.
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
#import "UIGamepadProvider.h"

#if ENABLE(GAMEPAD) && PLATFORM(IOS_FAMILY)

#import "UIKitSPI.h"
#import "WKWebViewInternal.h"
#import "WebPageProxy.h"

namespace WebKit {

WebPageProxy* UIGamepadProvider::platformWebPageProxyForGamepadInput()
{
ALLOW_DEPRECATED_DECLARATIONS_BEGIN
    auto firstResponder = [[[UIApplication sharedApplication] keyWindow] firstResponder];
ALLOW_DEPRECATED_DECLARATIONS_END

    if (auto *view = dynamic_objc_cast<WKContentView>(firstResponder))
        return view.page;

#if ENABLE(WEBXR) && !USE(OPENXR)
    if (auto page = WebProcessProxy::webPageWithActiveXRSession())
        return page.get();
#endif

    return nullptr;
}

}

#endif // ENABLE(GAMEPAD) && PLATFORM(IOS_FAMILY)
