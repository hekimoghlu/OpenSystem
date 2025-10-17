/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 21, 2022.
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

#if ENABLE(CONTENT_EXTENSIONS)

#include "ResourceMonitorChecker.h"
#include <wtf/CheckedArithmetic.h>
#include <wtf/RefCountedAndCanMakeWeakPtr.h>

namespace WebCore {

class LocalFrame;

class ResourceMonitor final : public RefCountedAndCanMakeWeakPtr<ResourceMonitor> {
public:
    using Eligibility = ResourceMonitorEligibility;

    static Ref<ResourceMonitor> create(LocalFrame&);

    Eligibility eligibility() const { return m_eligibility; }
    void setEligibility(Eligibility);

    void setDocumentURL(URL&&);
    void didReceiveResponse(const URL&, OptionSet<ContentExtensions::ResourceType>);
    void addNetworkUsage(size_t);

private:
    explicit ResourceMonitor(LocalFrame&);

    void checkNetworkUsageExcessIfNecessary();
    ResourceMonitor* parentResourceMonitorIfExists() const;

    WeakPtr<LocalFrame> m_frame;
    URL m_frameURL;
    Eligibility m_eligibility { Eligibility::Unsure };
    bool m_networkUsageExceed { false };
    CheckedSize m_networkUsage;
};

} // namespace WebCore

#endif
