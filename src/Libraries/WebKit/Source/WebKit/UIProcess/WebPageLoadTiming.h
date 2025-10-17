/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 25, 2021.
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

#include <wtf/FastMalloc.h>
#include <wtf/Noncopyable.h>
#include <wtf/WallTime.h>

namespace WebKit {

class WebPageLoadTiming {
    WTF_MAKE_NONCOPYABLE(WebPageLoadTiming);
    WTF_MAKE_FAST_ALLOCATED;
public:
    explicit WebPageLoadTiming(WallTime navigationStart)
        : m_navigationStart(navigationStart)
    { }

    WallTime navigationStart() const { return m_navigationStart; }

    WallTime firstVisualLayout() const { return m_firstVisualLayout; }
    void setFirstVisualLayout(WallTime timestamp) { m_firstVisualLayout = timestamp; }

    WallTime firstMeaningfulPaint() const { return m_firstMeaningfulPaint; }
    void setFirstMeaningfulPaint(WallTime timestamp) { m_firstMeaningfulPaint = timestamp; }

    WallTime documentFinishedLoading() const { return m_documentFinishedLoading; }
    void setDocumentFinishedLoading(WallTime timestamp) { m_documentFinishedLoading = timestamp; }

    WallTime allSubresourcesFinishedLoading() const { return m_allSubresourcesFinishedLoading; }
    void updateEndOfNetworkRequests(WallTime timestamp)
    {
        if (timestamp > m_allSubresourcesFinishedLoading)
            m_allSubresourcesFinishedLoading = timestamp;
    }

private:
    WallTime m_navigationStart;
    WallTime m_firstVisualLayout;
    WallTime m_firstMeaningfulPaint;
    WallTime m_documentFinishedLoading;
    WallTime m_allSubresourcesFinishedLoading;
};

}
