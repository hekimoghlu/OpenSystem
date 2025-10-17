/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 15, 2024.
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

#if ENABLE(CONTEXT_MENUS)

#include "APIClient.h"
#include "APIInjectedBundlePageContextMenuClient.h"
#include "WKBundlePageContextMenuClient.h"
#include <wtf/TZoneMalloc.h>

namespace API {
class Object;

template<> struct ClientTraits<WKBundlePageContextMenuClientBase> {
    typedef std::tuple<WKBundlePageContextMenuClientV0, WKBundlePageContextMenuClientV1> Versions;
};
}

namespace WebCore {
class ContextMenuContext;
class ContextMenuItem;
class HitTestResult;
}

namespace WebKit {
class WebContextMenuItemData;
class WebPage;

class InjectedBundlePageContextMenuClient : public API::Client<WKBundlePageContextMenuClientBase>, public API::InjectedBundle::PageContextMenuClient {
    WTF_MAKE_TZONE_ALLOCATED(InjectedBundlePageContextMenuClient);
public:
    explicit InjectedBundlePageContextMenuClient(const WKBundlePageContextMenuClientBase*);

private:
    bool getCustomMenuFromDefaultItems(WebPage&, const WebCore::HitTestResult&, const Vector<WebCore::ContextMenuItem>& defaultMenu, Vector<WebContextMenuItemData>& newMenu, const WebCore::ContextMenuContext&, RefPtr<API::Object>& userData) override;
    void prepareForImmediateAction(WebPage&, const WebCore::HitTestResult&, RefPtr<API::Object>& userData) override;
};

} // namespace WebKit

#endif // ENABLE(CONTEXT_MENUS)
