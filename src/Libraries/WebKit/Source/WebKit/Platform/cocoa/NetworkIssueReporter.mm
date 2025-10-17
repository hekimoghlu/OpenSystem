/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 16, 2021.
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
#import "NetworkIssueReporter.h"

#if ENABLE(NETWORK_ISSUE_REPORTING)

#import <pal/spi/cf/CFNetworkSPI.h>
#import <wtf/SoftLinking.h>
#import <wtf/TZoneMallocInlines.h>

SOFT_LINK_SYSTEM_LIBRARY(libsystem_networkextension)
SOFT_LINK_OPTIONAL(libsystem_networkextension, ne_tracker_create_xcode_issue, void, __cdecl, (const char*, const void*, size_t))
SOFT_LINK_OPTIONAL(libsystem_networkextension, ne_tracker_copy_current_stacktrace, void*, __cdecl, (size_t*))
SOFT_LINK_OPTIONAL(libsystem_networkextension, ne_tracker_should_save_stacktrace, bool, __cdecl, (void))

namespace WebKit {

WTF_MAKE_TZONE_ALLOCATED_IMPL(NetworkIssueReporter);

bool NetworkIssueReporter::isEnabled()
{
    auto* shouldSaveStacktrace = ne_tracker_should_save_stacktracePtr();
    return shouldSaveStacktrace && shouldSaveStacktrace();
}

bool NetworkIssueReporter::shouldReport(NSURLSessionTaskMetrics *metrics)
{
    if (!isEnabled())
        return false;

    for (NSURLSessionTaskTransactionMetrics *transaction in metrics.transactionMetrics) {
        if (transaction._isUnlistedTracker)
            return true;
    }

    return false;
}

NetworkIssueReporter::NetworkIssueReporter()
{
    if (auto* copyStacktrace = ne_tracker_copy_current_stacktracePtr())
        m_stackTrace = copyStacktrace(&m_stackTraceSize);
}

NetworkIssueReporter::~NetworkIssueReporter()
{
    if (m_stackTrace)
        free(m_stackTrace);
}

void NetworkIssueReporter::report(const URL& requestURL)
{
    if (!m_stackTrace)
        return;

    auto host = requestURL.host().toString();
    if (!m_reportedHosts.add(host).isNewEntry)
        return;

    if (auto createIssue = ne_tracker_create_xcode_issuePtr())
        createIssue(host.utf8().data(), m_stackTrace, m_stackTraceSize);
}

} // namespace WebKit

#endif // ENABLE(NETWORK_ISSUE_REPORTING)
