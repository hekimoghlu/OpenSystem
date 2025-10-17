/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 17, 2023.
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

#if USE(SYSTEM_PREVIEW)

#include "ProcessThrottler.h"
#include <WebCore/FrameLoaderTypes.h>
#include <WebCore/IntRect.h>
#include <WebCore/ResourceError.h>
#include <wtf/BlockPtr.h>
#include <wtf/RefCountedAndCanMakeWeakPtr.h>
#include <wtf/RetainPtr.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/URL.h>
#include <wtf/WeakPtr.h>

OBJC_CLASS NSArray;
OBJC_CLASS NSString;
#if USE(QUICK_LOOK)
OBJC_CLASS QLPreviewController;
OBJC_CLASS _WKPreviewControllerDataSource;
OBJC_CLASS _WKPreviewControllerDelegate;
OBJC_CLASS _WKSystemPreviewDataTaskDelegate;
#endif

namespace WebCore {
class SecurityOriginData;
}

namespace WebKit {

class WebPageProxy;

class SystemPreviewController : public RefCountedAndCanMakeWeakPtr<SystemPreviewController> {
    WTF_MAKE_TZONE_ALLOCATED(SystemPreviewController);
public:
    static Ref<SystemPreviewController> create(WebPageProxy&);

    bool canPreview(const String& mimeType) const;

    void begin(const URL&, const WebCore::SecurityOriginData& topOrigin, const WebCore::SystemPreviewInfo&, CompletionHandler<void()>&&);
    void updateProgress(float);
    void loadStarted(const URL& localFileURL);
    void loadCompleted(const URL& localFileURL);
    void loadFailed();
    void end();

    WebPageProxy* page() { return m_webPageProxy.get(); }
    const WebCore::SystemPreviewInfo& previewInfo() const { return m_systemPreviewInfo; }

    void triggerSystemPreviewAction();

    void triggerSystemPreviewActionWithTargetForTesting(uint64_t elementID, NSString* documentID, uint64_t pageID);
    void setCompletionHandlerForLoadTesting(CompletionHandler<void(bool)>&&);

private:
    explicit SystemPreviewController(WebPageProxy&);

    void takeActivityToken();
    void releaseActivityTokenIfNecessary();

    NSArray *localFileURLs() const;

    enum class State : uint8_t {
        Initial,
        Began,
        Loading,
        Viewing
    };

    State m_state { State::Initial };

    WeakPtr<WebPageProxy> m_webPageProxy;
    WebCore::SystemPreviewInfo m_systemPreviewInfo;
    URL m_downloadURL;
    URL m_localFileURL;
    String m_fragmentIdentifier;
#if USE(QUICK_LOOK)
    RetainPtr<QLPreviewController> m_qlPreviewController;
    RetainPtr<_WKPreviewControllerDelegate> m_qlPreviewControllerDelegate;
    RetainPtr<_WKPreviewControllerDataSource> m_qlPreviewControllerDataSource;
    RetainPtr<_WKSystemPreviewDataTaskDelegate> m_wkSystemPreviewDataTaskDelegate;
#endif

    RefPtr<ProcessThrottler::BackgroundActivity> m_activity;
    CompletionHandler<void(bool)> m_testingCallback;
    BlockPtr<void(bool)> m_allowPreviewCallback;
    double m_showPreviewDelay { 0 };

};

}

#endif
