/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 8, 2024.
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
#ifndef WebGeolocationProvider_h
#define WebGeolocationProvider_h

#include "APIClient.h"
#include "APIGeolocationProvider.h"
#include "WKGeolocationManager.h"
#include <wtf/Forward.h>
#include <wtf/TZoneMalloc.h>

namespace API {
template<> struct ClientTraits<WKGeolocationProviderBase> {
    typedef std::tuple<WKGeolocationProviderV0, WKGeolocationProviderV1> Versions;
};
}

namespace WebKit {

class WebGeolocationManagerProxy;

class WebGeolocationProvider : public API::GeolocationProvider, API::Client<WKGeolocationProviderBase> {
    WTF_MAKE_TZONE_ALLOCATED(WebGeolocationProvider);
public:
    explicit WebGeolocationProvider(const WKGeolocationProviderBase*);

    void startUpdating(WebGeolocationManagerProxy&) override;
    void stopUpdating(WebGeolocationManagerProxy&) override;
    void setEnableHighAccuracy(WebGeolocationManagerProxy&, bool) override;
};

} // namespace WebKit

#endif // WebGeolocationProvider_h
