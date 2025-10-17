/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 26, 2023.
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
#include "ITPThirdPartyData.h"
#include <wtf/RunLoop.h>
#include <wtf/text/WTFString.h>

namespace API {

class ResourceLoadStatisticsThirdParty final : public ObjectImpl<Object::Type::ResourceLoadStatisticsThirdParty> {
public:
    static Ref<ResourceLoadStatisticsThirdParty> create(WebKit::ITPThirdPartyData&& thirdPartyData)
    {
        RELEASE_ASSERT(RunLoop::isMain());
        return adoptRef(*new ResourceLoadStatisticsThirdParty(WTFMove(thirdPartyData)));
    }

    ~ResourceLoadStatisticsThirdParty()
    {
        RELEASE_ASSERT(RunLoop::isMain());
    }

    const WTF::String& thirdPartyDomain() const { return m_thirdPartyData.thirdPartyDomain.string(); }
    const Vector<WebKit::ITPThirdPartyDataForSpecificFirstParty>& underFirstParties() const { return m_thirdPartyData.underFirstParties; }

private:
    explicit ResourceLoadStatisticsThirdParty(WebKit::ITPThirdPartyData&& thirdPartyData)
        : m_thirdPartyData(WTFMove(thirdPartyData))
    {
    }

    const WebKit::ITPThirdPartyData m_thirdPartyData;
};

} // namespace API
