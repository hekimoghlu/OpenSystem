/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 25, 2022.
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
#include <wtf/RefPtr.h>
#include <wtf/text/WTFString.h>

namespace WebKit {
class WebProcessPool;
class WebURLSchemeHandler;
}

namespace API {

using URLSchemeHandlerPair = std::pair<Ref<WebKit::WebURLSchemeHandler>, WTF::String>;

class InspectorConfiguration final : public API::ObjectImpl<Object::Type::InspectorConfiguration> {
public:
    static Ref<InspectorConfiguration> create();

    InspectorConfiguration() = default;
    virtual ~InspectorConfiguration();

    void addURLSchemeHandler(Ref<WebKit::WebURLSchemeHandler>&&, const WTF::String& urlScheme);
    const Vector<URLSchemeHandlerPair>& urlSchemeHandlers() { return m_customURLSchemes; }
    
    WebKit::WebProcessPool* processPool();
    void setProcessPool(WebKit::WebProcessPool*);

private:
    Vector<URLSchemeHandlerPair> m_customURLSchemes;
    RefPtr<WebKit::WebProcessPool> m_processPool;
};

} // namespace API

SPECIALIZE_TYPE_TRAITS_API_OBJECT(InspectorConfiguration);
