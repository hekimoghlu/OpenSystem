/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 16, 2023.
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

#if ENABLE(GAMEPAD) && PLATFORM(MAC)

#import "WebPageProxy.h"
#import "WKAPICast.h"
#import "WKViewInternal.h"
#import "WKWebViewInternal.h"
#import <wtf/ProcessPrivilege.h>

namespace WebKit {

WebPageProxy* UIGamepadProvider::platformWebPageProxyForGamepadInput()
{
    ASSERT(hasProcessPrivilege(ProcessPrivilege::CanCommunicateWithWindowServer));
    auto responder = [[NSApp keyWindow] firstResponder];

    if (auto *view = dynamic_objc_cast<WKWebView>(responder))
        return view->_page.get();

ALLOW_DEPRECATED_DECLARATIONS_BEGIN
    if (auto *view = dynamic_objc_cast<WKView>(responder))
        return toImpl(view.pageRef);
ALLOW_DEPRECATED_DECLARATIONS_END

    return nullptr;
}

}

#endif // ENABLE(GAMEPAD) && PLATFORM(MAC)
