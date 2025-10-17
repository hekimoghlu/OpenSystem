/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 14, 2024.
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
#import "NetworkActivityTracker.h"

#if HAVE(NW_ACTIVITY)

namespace WebKit {

NetworkActivityTracker::NetworkActivityTracker(Label label, Domain domain)
    : m_domain(domain)
    , m_label(label)
    , m_networkActivity(adoptNS(nw_activity_create(static_cast<uint32_t>(m_domain), static_cast<uint32_t>(m_label))))
{
}

NetworkActivityTracker::~NetworkActivityTracker()
{
}

void NetworkActivityTracker::setParent(NetworkActivityTracker& parent)
{
    ASSERT(m_networkActivity.get());
    ASSERT(parent.m_networkActivity.get());
    nw_activity_set_parent_activity(m_networkActivity.get(), parent.m_networkActivity.get());
}

void NetworkActivityTracker::start()
{
    ASSERT(m_networkActivity.get());
    nw_activity_activate(m_networkActivity.get());
}

void NetworkActivityTracker::complete(CompletionCode code)
{
    if (m_isCompleted)
        return;

    m_isCompleted = true;

    ASSERT(m_networkActivity.get());
    nw_activity_completion_reason_t reason;
    switch (code) {
    case CompletionCode::Undefined:
        reason = nw_activity_completion_reason_invalid;
        break;
    case CompletionCode::None:
        reason = nw_activity_completion_reason_none;
        break;
    case CompletionCode::Success:
        reason = nw_activity_completion_reason_success;
        break;
    case CompletionCode::Failure:
        reason = nw_activity_completion_reason_failure;
        break;
    case CompletionCode::Cancel:
        reason = nw_activity_completion_reason_cancelled;
        break;
    }
    nw_activity_complete_with_reason(m_networkActivity.get(), reason);
    m_networkActivity.clear();
}

} // namespace WebKit

#endif // HAVE(NW_ACTIVITY)
