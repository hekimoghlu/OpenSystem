/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 24, 2023.
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

#include "APIClient.h"
#include "APIFormClient.h"
#include "WKPageFormClient.h"
#include <wtf/TZoneMalloc.h>

namespace API {
template<> struct ClientTraits<WKPageFormClientBase> {
    typedef std::tuple<WKPageFormClientV0> Versions;
};
}

namespace WebKit {

class WebFormClient : public API::FormClient, API::Client<WKPageFormClientBase> {
    WTF_MAKE_TZONE_ALLOCATED(WebFormClient);
public:
    explicit WebFormClient(const WKPageFormClientBase*);

    void willSubmitForm(WebPageProxy&, WebFrameProxy&, WebFrameProxy&, const Vector<std::pair<String, String>>& textFieldValues, API::Object* userData, CompletionHandler<void(void)>&&) override;
};

} // namespace WebKit
