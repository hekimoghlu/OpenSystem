/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 27, 2025.
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
#ifndef WebProgressTrackerClient_h
#define WebProgressTrackerClient_h

#include <WebCore/ProgressTrackerClient.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/WeakPtr.h>

namespace WebKit {

class WebPage;

class WebProgressTrackerClient : public WebCore::ProgressTrackerClient {
    WTF_MAKE_TZONE_ALLOCATED(WebProgressTrackerClient);
public:
    explicit WebProgressTrackerClient(WebPage&);
    
private:
    void progressStarted(WebCore::LocalFrame& originatingProgressFrame) override;
    void progressEstimateChanged(WebCore::LocalFrame& originatingProgressFrame) override;
    void progressFinished(WebCore::LocalFrame& originatingProgressFrame) override;

    WeakPtr<WebPage> m_webPage;
};

} // namespace WebKit

#endif // WebProgressTrackerClient_h
