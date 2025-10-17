/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 28, 2024.
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
#include "config.h"
#include "ResourceMonitor.h"

#include "Document.h"
#include "FrameLoader.h"
#include "HTMLIFrameElement.h"
#include "LocalDOMWindow.h"
#include "LocalFrame.h"
#include "LocalFrameLoaderClient.h"
#include "Logging.h"
#include "Page.h"
#include "ResourceMonitorChecker.h"
#include <wtf/StdLibExtras.h>

namespace WebCore {

#if ENABLE(CONTENT_EXTENSIONS)

#define RESOURCEMONITOR_RELEASE_LOG(fmt, ...) RELEASE_LOG(ResourceLoading, "%p - ResourceMonitor(frame %p)::" fmt, this, m_frame.get(), ##__VA_ARGS__)

Ref<ResourceMonitor> ResourceMonitor::create(LocalFrame& frame)
{
    return adoptRef(*new ResourceMonitor(frame));
}

ResourceMonitor::ResourceMonitor(LocalFrame& frame)
    : m_frame(frame)
{
    if (RefPtr parentMonitor = parentResourceMonitorIfExists())
        m_eligibility = parentMonitor->eligibility();
}

void ResourceMonitor::setEligibility(Eligibility eligibility)
{
    if (m_eligibility == eligibility || m_eligibility == Eligibility::Eligible)
        return;

    m_eligibility = eligibility;
    RESOURCEMONITOR_RELEASE_LOG("The frame is %" PUBLIC_LOG_STRING ".", (eligibility == Eligibility::Eligible ? "eligible" : "not eligible"));

    if (RefPtr parentMonitor = parentResourceMonitorIfExists())
        parentMonitor->setEligibility(eligibility);
    else
        checkNetworkUsageExcessIfNecessary();
}

void ResourceMonitor::setDocumentURL(URL&& url)
{
    RefPtr frame = m_frame.get();
    if (!frame)
        return;

    m_frameURL = WTFMove(url);

    didReceiveResponse(m_frameURL, ContentExtensions::ResourceType::Document);

    if (RefPtr iframe = dynamicDowncast<HTMLIFrameElement>(frame->ownerElement())) {
        if (auto& url = iframe->initiatorSourceURL(); !url.isEmpty())
            didReceiveResponse(url, ContentExtensions::ResourceType::Script);
    }
}

void ResourceMonitor::didReceiveResponse(const URL& url, OptionSet<ContentExtensions::ResourceType> resourceType)
{
    ASSERT(isMainThread());

    if (m_eligibility == Eligibility::Eligible)
        return;

    RefPtr frame = m_frame.get();
    RefPtr page = frame ? frame->mainFrame().page() : nullptr;
    if (!page)
        return;

    ContentExtensions::ResourceLoadInfo info = {
        .resourceURL = url,
        .mainDocumentURL = page->mainFrameURL(),
        .frameURL = m_frameURL,
        .type = resourceType
    };

    ResourceMonitorChecker::singleton().checkEligibility(WTFMove(info), [weakThis = WeakPtr { *this }](Eligibility eligibility) {
        if (RefPtr protectedThis = weakThis.get())
            protectedThis->setEligibility(eligibility);
    });
}

void ResourceMonitor::addNetworkUsage(size_t bytes)
{
    if (m_networkUsageExceed)
        return;

    m_networkUsage += bytes;

    if (RefPtr parentMonitor = parentResourceMonitorIfExists())
        parentMonitor->addNetworkUsage(bytes);
    else
        checkNetworkUsageExcessIfNecessary();
}

void ResourceMonitor::checkNetworkUsageExcessIfNecessary()
{
    ASSERT(!parentResourceMonitorIfExists());
    if (m_eligibility != Eligibility::Eligible || m_networkUsageExceed)
        return;

    if (m_networkUsage.hasOverflowed() || ResourceMonitorChecker::singleton().checkNetworkUsageExceedingThreshold(m_networkUsage)) {
        m_networkUsageExceed = true;

        RefPtr frame = m_frame.get();
        if (!frame)
            return;

        RESOURCEMONITOR_RELEASE_LOG("The frame exceeds the network usage threshold: used %ld", m_networkUsage.value());

        // If the frame has sticky user activation, don't do offloading.
        if (RefPtr protectedWindow = frame->window(); protectedWindow && protectedWindow->hasStickyActivation()) {
            RESOURCEMONITOR_RELEASE_LOG("But the frame has sticky user activation so ignoring.");
            return;
        }

        frame->loader().protectedClient()->didExceedNetworkUsageThreshold();
    }
}

ResourceMonitor* ResourceMonitor::parentResourceMonitorIfExists() const
{
    RefPtr frame = m_frame.get();
    RefPtr document = frame ? frame->document() : nullptr;
    return document ? document->parentResourceMonitorIfExists() : nullptr;
}

#undef RESOURCEMONITOR_RELEASE_LOG

#endif

} // namespace WebCore
