/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 15, 2025.
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

#include "APIObject.h"
#include "ContentWorldData.h"
#include <wtf/WeakHashSet.h>
#include <wtf/WeakPtr.h>
#include <wtf/text/WTFString.h>

namespace WebKit {
class WebUserContentControllerProxy;
}

namespace API {

class ContentWorld final : public API::ObjectImpl<API::Object::Type::ContentWorld>, public CanMakeWeakPtr<ContentWorld> {
public:
    static ContentWorld* worldForIdentifier(WebKit::ContentWorldIdentifier);
    static Ref<ContentWorld> sharedWorldWithName(const WTF::String&, OptionSet<WebKit::ContentWorldOption> options = { });
    static ContentWorld& pageContentWorldSingleton();
    static ContentWorld& defaultClientWorldSingleton();

    virtual ~ContentWorld();

    WebKit::ContentWorldIdentifier identifier() const { return m_identifier; }
    const WTF::String& name() const { return m_name; }
    WebKit::ContentWorldData worldData() const { return { m_identifier, m_name, m_options }; }

    bool allowAccessToClosedShadowRoots() const { return m_options.contains(WebKit::ContentWorldOption::AllowAccessToClosedShadowRoots); }
    void setAllowAccessToClosedShadowRoots(bool value) { m_options.add(WebKit::ContentWorldOption::AllowAccessToClosedShadowRoots); }

    bool allowAutofill() const { return m_options.contains(WebKit::ContentWorldOption::AllowAutofill); }
    void setAllowAutofill(bool value) { m_options.add(WebKit::ContentWorldOption::AllowAutofill); }

    bool allowElementUserInfo() const { return m_options.contains(WebKit::ContentWorldOption::AllowElementUserInfo); }
    void setAllowElementUserInfo(bool value) { m_options.add(WebKit::ContentWorldOption::AllowElementUserInfo); }

    bool disableLegacyBuiltinOverrides() const { return m_options.contains(WebKit::ContentWorldOption::DisableLegacyBuiltinOverrides); }
    void setDisableLegacyBuiltinOverrides(bool value) { m_options.add(WebKit::ContentWorldOption::DisableLegacyBuiltinOverrides); }

    void addAssociatedUserContentControllerProxy(WebKit::WebUserContentControllerProxy&);
    void userContentControllerProxyDestroyed(WebKit::WebUserContentControllerProxy&);

private:
    explicit ContentWorld(const WTF::String&, OptionSet<WebKit::ContentWorldOption>);
    explicit ContentWorld(WebKit::ContentWorldIdentifier);

    WebKit::ContentWorldIdentifier m_identifier;
    WTF::String m_name;
    OptionSet<WebKit::ContentWorldOption> m_options;
    WeakHashSet<WebKit::WebUserContentControllerProxy> m_associatedContentControllerProxies;
};

} // namespace API
