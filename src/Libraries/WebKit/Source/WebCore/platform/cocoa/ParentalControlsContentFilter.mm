/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 10, 2023.
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
#import "config.h"
#import "ParentalControlsContentFilter.h"

#if HAVE(PARENTAL_CONTROLS)

#import "ContentFilterUnblockHandler.h"
#import "Logging.h"
#import "ResourceResponse.h"
#import "SharedBuffer.h"
#import <objc/runtime.h>
#import <pal/spi/cocoa/WebFilterEvaluatorSPI.h>
#import <wtf/SoftLinking.h>
#import <wtf/TZoneMallocInlines.h>

SOFT_LINK_PRIVATE_FRAMEWORK_OPTIONAL(WebContentAnalysis);
SOFT_LINK_CLASS_OPTIONAL(WebContentAnalysis, WebFilterEvaluator);

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(ParentalControlsContentFilter);

bool ParentalControlsContentFilter::enabled()
{
    bool enabled = [getWebFilterEvaluatorClass() isManagedSession];
    LOG(ContentFiltering, "ParentalControlsContentFilter is %s.\n", enabled ? "enabled" : "not enabled");
    return enabled;
}

UniqueRef<ParentalControlsContentFilter> ParentalControlsContentFilter::create()
{
    return makeUniqueRef<ParentalControlsContentFilter>();
}

static inline bool canHandleResponse(const ResourceResponse& response)
{
#if HAVE(SYSTEM_HTTP_CONTENT_FILTERING)
    return response.url().protocolIs("https"_s);
#else
    return response.url().protocolIsInHTTPFamily();
#endif
}

void ParentalControlsContentFilter::responseReceived(const ResourceResponse& response)
{
    ASSERT(!m_webFilterEvaluator);

    if (!canHandleResponse(response) || !enabled()) {
        m_state = State::Allowed;
        return;
    }

    m_webFilterEvaluator = adoptNS([allocWebFilterEvaluatorInstance() initWithResponse:response.nsURLResponse()]);
#if HAVE(WEBFILTEREVALUATOR_AUDIT_TOKEN)
    if (m_hostProcessAuditToken)
        m_webFilterEvaluator.get().browserAuditToken = *m_hostProcessAuditToken;
#endif
    updateFilterState();
}

void ParentalControlsContentFilter::addData(const SharedBuffer& data)
{
    ASSERT(![m_replacementData length]);
    m_replacementData = [m_webFilterEvaluator addData:data.createNSData().get()];
    updateFilterState();
    ASSERT(needsMoreData() || [m_replacementData length]);
}

void ParentalControlsContentFilter::finishedAddingData()
{
    ASSERT(![m_replacementData length]);
    m_replacementData = [m_webFilterEvaluator dataComplete];
    updateFilterState();
}

Ref<FragmentedSharedBuffer> ParentalControlsContentFilter::replacementData() const
{
    ASSERT(didBlockData());
    return SharedBuffer::create(m_replacementData.get());
}

#if ENABLE(CONTENT_FILTERING)
ContentFilterUnblockHandler ParentalControlsContentFilter::unblockHandler() const
{
#if HAVE(PARENTAL_CONTROLS_WITH_UNBLOCK_HANDLER)
    return ContentFilterUnblockHandler { "unblock"_s, m_webFilterEvaluator };
#else
    return { };
#endif
}
#endif

void ParentalControlsContentFilter::updateFilterState()
{
    switch ([m_webFilterEvaluator filterState]) {
    case kWFEStateAllowed:
    case kWFEStateEvaluating:
        m_state = State::Allowed;
        break;
    case kWFEStateBlocked:
        m_state = State::Blocked;
        break;
    case kWFEStateBuffering:
        m_state = State::Filtering;
        break;
    }

#if !LOG_DISABLED
    if (!needsMoreData())
        LOG(ContentFiltering, "ParentalControlsContentFilter stopped buffering with state %d and replacement data length %zu.\n", m_state, [m_replacementData length]);
#endif
}

} // namespace WebCore

#endif // HAVE(PARENTAL_CONTROLS)
