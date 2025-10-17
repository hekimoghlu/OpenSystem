/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 1, 2022.
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
#ifndef InjectedBundleScriptWorld_h
#define InjectedBundleScriptWorld_h

#include "APIObject.h"
#include <wtf/Ref.h>
#include <wtf/RefPtr.h>
#include <wtf/WeakPtr.h>
#include <wtf/text/WTFString.h>

namespace WebCore {
class DOMWrapperWorld;
}

namespace WebKit {

class InjectedBundleScriptWorld : public API::ObjectImpl<API::Object::Type::BundleScriptWorld>, public CanMakeWeakPtr<InjectedBundleScriptWorld> {
public:
    enum class Type { User, Internal };
    static Ref<InjectedBundleScriptWorld> create(Type = Type::Internal);
    static Ref<InjectedBundleScriptWorld> create(const String& name, Type = Type::Internal);
    static Ref<InjectedBundleScriptWorld> getOrCreate(WebCore::DOMWrapperWorld&);
    static InjectedBundleScriptWorld* find(const String&);
    static InjectedBundleScriptWorld& normalWorld();

    virtual ~InjectedBundleScriptWorld();

    const WebCore::DOMWrapperWorld& coreWorld() const;
    WebCore::DOMWrapperWorld& coreWorld();

    void clearWrappers();
    void setAllowAutofill();
    void setAllowElementUserInfo();
    void makeAllShadowRootsOpen();
    void disableOverrideBuiltinsBehavior();

    const String& name() const { return m_name; }

private:
    InjectedBundleScriptWorld(WebCore::DOMWrapperWorld&, const String&);

    Ref<WebCore::DOMWrapperWorld> m_world;
    String m_name;
};

} // namespace WebKit

#endif // InjectedBundleScriptWorld_h
