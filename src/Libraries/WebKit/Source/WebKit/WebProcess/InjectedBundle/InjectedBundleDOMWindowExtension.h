/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 15, 2024.
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
#ifndef InjectedBundleDOMWindowExtension_h
#define InjectedBundleDOMWindowExtension_h

#include "APIObject.h"
#include <wtf/RefPtr.h>
#include <wtf/WeakPtr.h>

namespace WebCore {

class DOMWindowExtension;

}

namespace WebKit {

class InjectedBundleScriptWorld;
class WebFrame;

class InjectedBundleDOMWindowExtension : public API::ObjectImpl<API::Object::Type::BundleDOMWindowExtension>, public CanMakeWeakPtr<InjectedBundleDOMWindowExtension> {
public:
    static Ref<InjectedBundleDOMWindowExtension> create(WebFrame*, InjectedBundleScriptWorld*);
    static InjectedBundleDOMWindowExtension* get(WebCore::DOMWindowExtension*);

    virtual ~InjectedBundleDOMWindowExtension();
    
    RefPtr<WebFrame> frame() const;
    InjectedBundleScriptWorld* world() const;

private:
    InjectedBundleDOMWindowExtension(WebFrame*, InjectedBundleScriptWorld*);

    Ref<WebCore::DOMWindowExtension> m_coreExtension;
    mutable RefPtr<InjectedBundleScriptWorld> m_world;
};

} // namespace WebKit

#endif // InjectedBundleDOMWindowExtension_h
