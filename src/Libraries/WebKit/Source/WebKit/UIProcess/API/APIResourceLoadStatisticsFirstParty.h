/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 12, 2023.
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
#include "ITPThirdPartyDataForSpecificFirstParty.h"
#include <wtf/RunLoop.h>
#include <wtf/text/WTFString.h>

namespace API {

class ResourceLoadStatisticsFirstParty final : public ObjectImpl<Object::Type::ResourceLoadStatisticsFirstParty> {
public:
    static Ref<ResourceLoadStatisticsFirstParty> create(const WebKit::ITPThirdPartyDataForSpecificFirstParty& firstPartyData)
    {
        RELEASE_ASSERT(RunLoop::isMain());
        return adoptRef(*new ResourceLoadStatisticsFirstParty(firstPartyData));
    }

    ~ResourceLoadStatisticsFirstParty()
    {
        RELEASE_ASSERT(RunLoop::isMain());
    }

    const WTF::String& firstPartyDomain() const { return m_firstPartyData.firstPartyDomain.string(); }
    bool storageAccess() const { return m_firstPartyData.storageAccessGranted; }
    double timeLastUpdated() const { return m_firstPartyData.timeLastUpdated.value(); }

private:
    explicit ResourceLoadStatisticsFirstParty(const WebKit::ITPThirdPartyDataForSpecificFirstParty& firstPartyData)
        : m_firstPartyData(firstPartyData)
    {
    }

    const WebKit::ITPThirdPartyDataForSpecificFirstParty m_firstPartyData;
};

} // namespace API
