/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 10, 2025.
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

#if PLATFORM(IOS_FAMILY)

#import <wtf/Forward.h>
#import <wtf/RefCountedAndCanMakeWeakPtr.h>
#import <wtf/TZoneMalloc.h>
#import <wtf/WeakObjCPtr.h>
#import <wtf/WeakPtr.h>

OBJC_CLASS BKSApplicationStateMonitor;
OBJC_CLASS UIView;
OBJC_CLASS UIViewController;
OBJC_CLASS UIWindow;
OBJC_CLASS UIScene;
OBJC_CLASS WKUIWindowSceneObserver;

namespace WebKit {

enum class ApplicationType : uint8_t {
    Application,
    ViewService,
    Extension,
};

class ApplicationStateTracker : public RefCountedAndCanMakeWeakPtr<ApplicationStateTracker> {
    WTF_MAKE_TZONE_ALLOCATED(ApplicationStateTracker);
public:
    static RefPtr<ApplicationStateTracker> create(UIView *view, SEL didEnterBackgroundSelector, SEL willEnterForegroundSelector, SEL willBeginSnapshotSequenceSelector, SEL didCompleteSnapshotSequenceSelector)
    {
        return adoptRef(new ApplicationStateTracker(view, didEnterBackgroundSelector, willEnterForegroundSelector, willBeginSnapshotSequenceSelector, didCompleteSnapshotSequenceSelector));
    }

    ~ApplicationStateTracker();

    bool isInBackground() const { return m_isInBackground; }

    void setWindow(UIWindow *);
    void setScene(UIScene *);

private:
    ApplicationStateTracker(UIView *, SEL didEnterBackgroundSelector, SEL willEnterForegroundSelector, SEL willBeginSnapshotSequenceSelector, SEL didCompleteSnapshotSequenceSelector);

    void setViewController(UIViewController *);

    void applicationDidEnterBackground();
    void applicationDidFinishSnapshottingAfterEnteringBackground();
    void applicationWillEnterForeground();
    void willBeginSnapshotSequence();
    void didCompleteSnapshotSequence();
    void removeAllObservers();

    WeakObjCPtr<UIView> m_view;
    WeakObjCPtr<UIWindow> m_window;
    WeakObjCPtr<UIScene> m_scene;
    WeakObjCPtr<UIViewController> m_viewController;

    RetainPtr<WKUIWindowSceneObserver> m_observer;

    ApplicationType m_applicationType { ApplicationType::Application };

    SEL m_didEnterBackgroundSelector;
    SEL m_willEnterForegroundSelector;
    SEL m_willBeginSnapshotSequenceSelector;
    SEL m_didCompleteSnapshotSequenceSelector;

    bool m_isInBackground;

    WeakObjCPtr<NSObject> m_didEnterBackgroundObserver;
    WeakObjCPtr<NSObject> m_willEnterForegroundObserver;
    WeakObjCPtr<NSObject> m_willBeginSnapshotSequenceObserver;
    WeakObjCPtr<NSObject> m_didCompleteSnapshotSequenceObserver;
};

ApplicationType applicationType(UIWindow *);

}

#endif
