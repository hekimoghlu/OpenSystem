/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 19, 2024.
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
#ifndef InjectedBundleClient_h
#define InjectedBundleClient_h

#include "APIClient.h"
#include "APIInjectedBundleBundleClient.h"
#include "WKBundle.h"
#include <wtf/Forward.h>
#include <wtf/TZoneMalloc.h>

namespace API {
class Object;

template<> struct ClientTraits<WKBundleClientBase> {
    typedef std::tuple<WKBundleClientV0, WKBundleClientV1> Versions;
};
}

namespace WebKit {

class InjectedBundle;
class WebPage;
class WebPageGroupProxy;

class InjectedBundleClient : public API::InjectedBundle::Client, public API::Client<WKBundleClientBase> {
    WTF_MAKE_TZONE_ALLOCATED(InjectedBundleClient);
public:
    explicit InjectedBundleClient(const WKBundleClientBase*);

    void didCreatePage(InjectedBundle&, WebPage&) override;
    void willDestroyPage(InjectedBundle&, WebPage&) override;
    void didReceiveMessage(InjectedBundle&, const WTF::String&, RefPtr<API::Object>&&) override;
    void didReceiveMessageToPage(InjectedBundle&, WebPage&, const WTF::String&, RefPtr<API::Object>&&) override;
};

} // namespace WebKit

#endif // InjectedBundleClient_h
