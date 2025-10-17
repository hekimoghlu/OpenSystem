/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 19, 2021.
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
#import "_WKWebExtensionSidebar.h"

#if ENABLE(WK_WEB_EXTENSIONS_SIDEBAR)

#import "WKObject.h"
#import "WebExtensionSidebar.h"

namespace WebKit {
template<> struct WrapperTraits<WebExtensionSidebar> {
    using WrapperClass = _WKWebExtensionSidebar;
};
}

@interface _WKWebExtensionSidebar () <WKObject> {
@package
    API::ObjectStorage<WebKit::WebExtensionSidebar> _webExtensionSidebar;
}

@property (nonatomic, readonly) WebKit::WebExtensionSidebar& _webExtensionSidebar;

@end

#endif // ENABLE(WK_WEB_EXTENSIONS_SIDEBAR)
