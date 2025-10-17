/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 13, 2021.
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
#if !__has_feature(objc_arc)
#error This file requires ARC. Add the "-fobjc-arc" compiler flag for this file.
#endif

#import "config.h"
#import "WebExtensionAPIWebPageNamespace.h"

#if ENABLE(WK_WEB_EXTENSIONS)

#import "WebExtensionAPINamespace.h"
#import "WebExtensionAPIRuntime.h"
#import "WebExtensionAPITest.h"
#import "WebExtensionContextProxy.h"
#import "WebExtensionControllerProxy.h"
#import "WebPage.h"

namespace WebKit {

bool WebExtensionAPIWebPageNamespace::isPropertyAllowed(const ASCIILiteral& name, WebPage* page)
{
    if (name == "test"_s) {
        if (!page)
            return false;
        if (RefPtr extensionController = page->webExtensionControllerProxy())
            return extensionController->inTestingMode();
        return false;
    }

    ASSERT_NOT_REACHED();
    return false;
}

WebExtensionAPIWebPageRuntime& WebExtensionAPIWebPageNamespace::runtime() const
{
    // Documentation: https://developer.mozilla.org/docs/Mozilla/Add-ons/WebExtensions/API/runtime

    if (!m_runtime) {
        m_runtime = WebExtensionAPIWebPageRuntime::create(contentWorldType());
        m_runtime->setPropertyPath("runtime"_s, this);
    }

    return *m_runtime;
}

WebExtensionAPITest& WebExtensionAPIWebPageNamespace::test()
{
    // Documentation: None (Testing Only)

    if (!m_test)
        m_test = WebExtensionAPITest::create(*this);

    return *m_test;
}

} // namespace WebKit

#endif // ENABLE(WK_WEB_EXTENSIONS)
