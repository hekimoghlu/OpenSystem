/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 14, 2023.
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

#include <WebCore/PerformanceLoggingClient.h>
#include <wtf/Forward.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/WeakRef.h>

namespace WebKit {

class WebPage;

class WebPerformanceLoggingClient : public WebCore::PerformanceLoggingClient {
    WTF_MAKE_TZONE_ALLOCATED(WebPerformanceLoggingClient);
public:
    WebPerformanceLoggingClient(WebPage&);
    virtual ~WebPerformanceLoggingClient() = default;

private:
    void logScrollingEvent(ScrollingEvent, MonotonicTime, uint64_t) override;

    WeakRef<WebPage> m_page;
};

}
