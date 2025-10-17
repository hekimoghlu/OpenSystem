/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 10, 2023.
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
#include "WebKitInjectedBundleClient.h"

#include "APIInjectedBundleClient.h"
#include "APIString.h"
#include "WebImage.h"
#include "WebKitPrivate.h"
#include "WebKitURIRequestPrivate.h"
#include "WebKitURIResponsePrivate.h"
#include "WebKitWebContextPrivate.h"
#include "WebKitWebResourcePrivate.h"
#include "WebKitWebViewPrivate.h"
#include <wtf/glib/GUniquePtr.h>

using namespace WebKit;
using namespace WebCore;

class WebKitInjectedBundleClient final : public API::InjectedBundleClient {
public:
    explicit WebKitInjectedBundleClient(WebKitWebContext* webContext)
        : m_webContext(webContext)
    {
    }

private:
    RefPtr<API::Object> getInjectedBundleInitializationUserData(WebProcessPool&) override
    {
        GRefPtr<GVariant> data = webkitWebContextInitializeWebProcessExtensions(m_webContext);
        GUniquePtr<gchar> dataString(g_variant_print(data.get(), TRUE));
        return API::String::create(String::fromUTF8(dataString.get()));
    }

    WebKitWebContext* m_webContext;
};

void attachInjectedBundleClientToContext(WebKitWebContext* webContext)
{
    webkitWebContextGetProcessPool(webContext).setInjectedBundleClient(makeUnique<WebKitInjectedBundleClient>(webContext));
}
