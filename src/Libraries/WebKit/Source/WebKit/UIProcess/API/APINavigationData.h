/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 26, 2022.
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
#ifndef APINavigationData_h
#define APINavigationData_h

#include "APIObject.h"
#include "WebNavigationDataStore.h"

namespace API {

class NavigationData : public ObjectImpl<Object::Type::NavigationData> {
public:
    static Ref<NavigationData> create(const WebKit::WebNavigationDataStore& store)
    {
        return adoptRef(*new NavigationData(store));
    }

    virtual ~NavigationData();

    WTF::String title() const { return m_store.title; }
    WTF::String url() const { return m_store.url; }
    const WebCore::ResourceRequest& originalRequest() const { return m_store.originalRequest; }
    const WebCore::ResourceResponse& response() const { return m_store.response; }

private:
    explicit NavigationData(const WebKit::WebNavigationDataStore&);

    WebKit::WebNavigationDataStore m_store;
};

} // namespace API

#endif // APINavigationData_h
