/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 20, 2021.
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
#include <wtf/CompletionHandler.h>
#include <wtf/Forward.h>
#include <wtf/RefPtr.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebKit {
class WebProcessPool;
}

namespace API {

class InjectedBundleClient {
    WTF_MAKE_TZONE_ALLOCATED_INLINE(InjectedBundleClient);
public:
    virtual ~InjectedBundleClient() = default;

    virtual void didReceiveMessageFromInjectedBundle(WebKit::WebProcessPool&, const WTF::String&, API::Object*) { }
    virtual void didReceiveSynchronousMessageFromInjectedBundle(WebKit::WebProcessPool&, const WTF::String&, API::Object*, CompletionHandler<void(RefPtr<API::Object>)>&& completionHandler) { completionHandler(nullptr); }
    virtual RefPtr<API::Object> getInjectedBundleInitializationUserData(WebKit::WebProcessPool&) { return nullptr; }
};

} // namespace API
