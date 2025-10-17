/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 5, 2024.
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
#include "MockContentFilter.h"

#if ENABLE(CONTENT_FILTERING)

#include "ContentFilter.h"
#include "ContentFilterUnblockHandler.h"
#include "Logging.h"
#include "ResourceRequest.h"
#include "ResourceResponse.h"
#include "SharedBuffer.h"
#include <mutex>
#include <wtf/EnumTraits.h>
#include <wtf/TZoneMallocInlines.h>
#include <wtf/text/CString.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(MockContentFilter);

using Decision = MockContentFilterSettings::Decision;
using DecisionPoint = MockContentFilterSettings::DecisionPoint;

void MockContentFilter::ensureInstalled()
{
    static std::once_flag onceFlag;
    std::call_once(onceFlag, []{
        ContentFilter::addType<MockContentFilter>();
    });
}

static inline MockContentFilterSettings& settings()
{
    return MockContentFilterSettings::singleton();
}

bool MockContentFilter::enabled()
{
    bool enabled = settings().enabled();
    LOG(ContentFiltering, "MockContentFilter is %s.\n", enabled ? "enabled" : "not enabled");
    return enabled;
}

UniqueRef<MockContentFilter> MockContentFilter::create()
{
    return makeUniqueRef<MockContentFilter>();
}

void MockContentFilter::willSendRequest(ResourceRequest& request, const ResourceResponse& redirectResponse)
{
    if (!enabled()) {
        m_state = State::Allowed;
        return;
    }

    if (redirectResponse.isNull())
        maybeDetermineStatus(DecisionPoint::AfterWillSendRequest);
    else
        maybeDetermineStatus(DecisionPoint::AfterRedirect);

    if (m_state == State::Filtering)
        return;

    String modifiedRequestURLString { settings().modifiedRequestURL() };
    if (modifiedRequestURLString.isEmpty())
        return;

    URL modifiedRequestURL { request.url(), modifiedRequestURLString };
    if (!modifiedRequestURL.isValid()) {
        LOG(ContentFiltering, "MockContentFilter failed to convert %s to a  URL.\n", modifiedRequestURL.string().ascii().data());
        return;
    }

    request.setURL(modifiedRequestURL);
}

void MockContentFilter::responseReceived(const ResourceResponse&)
{
    maybeDetermineStatus(DecisionPoint::AfterResponse);
}

void MockContentFilter::addData(const SharedBuffer&)
{
    maybeDetermineStatus(DecisionPoint::AfterAddData);
}

void MockContentFilter::finishedAddingData()
{
    maybeDetermineStatus(DecisionPoint::AfterFinishedAddingData);
}

Ref<FragmentedSharedBuffer> MockContentFilter::replacementData() const
{
    ASSERT(didBlockData());
    return SharedBuffer::create(m_replacementData.span());
}

ContentFilterUnblockHandler MockContentFilter::unblockHandler() const
{
    ASSERT(didBlockData());
    using DecisionHandlerFunction = ContentFilterUnblockHandler::DecisionHandlerFunction;

    return ContentFilterUnblockHandler {
        MockContentFilterSettings::unblockURLHost(), [](DecisionHandlerFunction decisionHandler) {
            bool shouldAllow { settings().unblockRequestDecision() == Decision::Allow };
            if (shouldAllow)
                settings().setDecision(Decision::Allow);
            LOG(ContentFiltering, "MockContentFilter %s the unblock request.\n", shouldAllow ? "allowed" : "did not allow");
            decisionHandler(shouldAllow);
        }
    };
}

String MockContentFilter::unblockRequestDeniedScript() const
{
    return "unblockRequestDenied()"_s;
}

void MockContentFilter::maybeDetermineStatus(DecisionPoint decisionPoint)
{
    if (m_state != State::Filtering || decisionPoint != settings().decisionPoint())
        return;

    LOG(ContentFiltering, "MockContentFilter stopped buffering with state %u at decision point %hhu.\n", enumToUnderlyingType(m_state), enumToUnderlyingType(decisionPoint));

    m_state = settings().decision() == Decision::Allow ? State::Allowed : State::Blocked;
    if (m_state != State::Blocked)
        return;

    m_replacementData = settings().blockedString().utf8().span();
}

} // namespace WebCore

#endif // ENABLE(CONTENT_FILTERING)
