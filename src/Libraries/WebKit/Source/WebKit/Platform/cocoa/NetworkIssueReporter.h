/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 15, 2022.
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

#if ENABLE(NETWORK_ISSUE_REPORTING)

#import <wtf/Forward.h>
#import <wtf/TZoneMalloc.h>

OBJC_CLASS NSURLSessionTaskMetrics;

namespace WebKit {

class NetworkIssueReporter {
    WTF_MAKE_TZONE_ALLOCATED(NetworkIssueReporter);
    WTF_MAKE_NONCOPYABLE(NetworkIssueReporter);
public:
    NetworkIssueReporter();
    ~NetworkIssueReporter();

    void report(const URL&);

    static bool isEnabled();
    static bool shouldReport(NSURLSessionTaskMetrics *);

private:
    HashSet<String> m_reportedHosts;
    void* m_stackTrace { nullptr };
    size_t m_stackTraceSize { 0 };
};

} // namespace WebKit

#endif // ENABLE(NETWORK_ISSUE_REPORTING)
