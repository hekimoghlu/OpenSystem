/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 2, 2023.
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

#include <WebCore/NavigationIdentifier.h>
#include <WebCore/ProcessIdentifier.h>
#include <wtf/HashMap.h>
#include <wtf/Ref.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/WeakPtr.h>

namespace API {
class Navigation;
struct SubstituteData;
}

namespace WebCore {
class ResourceRequest;

enum class FrameLoadType : uint8_t;
}

namespace WebKit {

class WebPageProxy;
class WebBackForwardListFrameItem;
class WebBackForwardListItem;

class WebNavigationState : public CanMakeWeakPtr<WebNavigationState> {
    WTF_MAKE_TZONE_ALLOCATED(WebNavigationState);
public:
    explicit WebNavigationState(WebPageProxy&);
    ~WebNavigationState();

    void ref() const;
    void deref() const;

    Ref<API::Navigation> createBackForwardNavigation(WebCore::ProcessIdentifier, Ref<WebBackForwardListFrameItem>&& targetFrameItem, RefPtr<WebBackForwardListItem>&& currentItem, WebCore::FrameLoadType);
    Ref<API::Navigation> createLoadRequestNavigation(WebCore::ProcessIdentifier, WebCore::ResourceRequest&&, RefPtr<WebBackForwardListItem>&& currentItem);
    Ref<API::Navigation> createReloadNavigation(WebCore::ProcessIdentifier, RefPtr<WebBackForwardListItem>&& currentAndTargetItem);
    Ref<API::Navigation> createLoadDataNavigation(WebCore::ProcessIdentifier, std::unique_ptr<API::SubstituteData>&&);
    Ref<API::Navigation> createSimulatedLoadWithDataNavigation(WebCore::ProcessIdentifier, WebCore::ResourceRequest&&, std::unique_ptr<API::SubstituteData>&&, RefPtr<WebBackForwardListItem>&& currentItem);

    bool hasNavigation(WebCore::NavigationIdentifier navigationID) const { return m_navigations.contains(navigationID); }
    API::Navigation* navigation(WebCore::NavigationIdentifier);
    RefPtr<API::Navigation> takeNavigation(WebCore::NavigationIdentifier);
    void didDestroyNavigation(WebCore::ProcessIdentifier, WebCore::NavigationIdentifier);
    void clearAllNavigations();

    void clearNavigationsFromProcess(WebCore::ProcessIdentifier);

    using NavigationMap = HashMap<WebCore::NavigationIdentifier, RefPtr<API::Navigation>>;

private:
    WeakRef<WebPageProxy> m_page;
    NavigationMap m_navigations;
};

} // namespace WebKit
