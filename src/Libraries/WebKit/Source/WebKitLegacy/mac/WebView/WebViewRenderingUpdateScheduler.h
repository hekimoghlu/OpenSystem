/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 19, 2024.
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
#import <WebCore/RunLoopObserver.h>
#import <wtf/CheckedPtr.h>
#import <wtf/TZoneMalloc.h>
#import <wtf/WeakPtr.h>

@class WebView;

class WebViewRenderingUpdateScheduler : public CanMakeWeakPtr<WebViewRenderingUpdateScheduler>, public CanMakeCheckedPtr<WebViewRenderingUpdateScheduler> {
    WTF_MAKE_TZONE_ALLOCATED(WebViewRenderingUpdateScheduler);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(WebViewRenderingUpdateScheduler);
public:
    explicit WebViewRenderingUpdateScheduler(WebView*);
    ~WebViewRenderingUpdateScheduler();

    void scheduleRenderingUpdate();
    void invalidate();

    void didCompleteRenderingUpdateDisplay();
    
private:
    void registerCACommitHandlers();

    void renderingUpdateRunLoopObserverCallback();
    void updateRendering();

    void schedulePostRenderingUpdate();
    void postRenderingUpdateCallback();

    WebView* m_webView;

    std::unique_ptr<WebCore::RunLoopObserver> m_renderingUpdateRunLoopObserver;
    std::unique_ptr<WebCore::RunLoopObserver> m_postRenderingUpdateRunLoopObserver;

    bool m_insideCallback { false };
    bool m_rescheduledInsideCallback { false };
    bool m_haveRegisteredCommitHandlers { false };
};
