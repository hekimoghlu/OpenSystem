/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 11, 2022.
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

#if HAVE(SCREEN_CAPTURE_KIT)

#include "DisplayCapturePromptType.h"
#include <wtf/CompletionHandler.h>
#include <wtf/RetainPtr.h>
#include <wtf/RunLoop.h>
#include <wtf/WeakPtr.h>

OBJC_CLASS NSError;
OBJC_CLASS SCContentFilter;
OBJC_CLASS SCContentSharingPicker;
OBJC_CLASS SCContentSharingSession;
OBJC_CLASS SCStream;
OBJC_CLASS SCStreamConfiguration;
OBJC_CLASS SCStreamDelegate;
OBJC_CLASS WebDisplayMediaPromptHelper;

namespace WebCore {
class ScreenCaptureKitSharingSessionManager;
class ScreenCaptureSessionSourceObserver;
}

namespace WTF {
template<typename T> struct IsDeprecatedWeakRefSmartPointerException;
template<> struct IsDeprecatedWeakRefSmartPointerException<WebCore::ScreenCaptureKitSharingSessionManager> : std::true_type { };
template<> struct IsDeprecatedWeakRefSmartPointerException<WebCore::ScreenCaptureSessionSourceObserver> : std::true_type { };
}

namespace WebCore {

class CaptureDevice;
class ScreenCaptureKitSharingSessionManager;

class ScreenCaptureSessionSourceObserver : public CanMakeWeakPtr<ScreenCaptureSessionSourceObserver> {
public:
    virtual ~ScreenCaptureSessionSourceObserver() = default;

    // Session state changes.
    virtual void sessionFilterDidChange(SCContentFilter*) = 0;
    virtual void sessionStreamDidEnd(SCStream*) = 0;
};

class ScreenCaptureSessionSource
    : public RefCounted<ScreenCaptureSessionSource>
    , public CanMakeWeakPtr<ScreenCaptureSessionSource> {
public:
    using CleanupFunction = CompletionHandler<void(ScreenCaptureSessionSource&)>;
    static Ref<ScreenCaptureSessionSource> create(WeakPtr<ScreenCaptureSessionSourceObserver>, RetainPtr<SCStream>, RetainPtr<SCContentFilter>, RetainPtr<SCContentSharingSession>, CleanupFunction&&);
    virtual ~ScreenCaptureSessionSource();

    SCStream* stream() const { return m_stream.get(); }
    SCContentFilter* contentFilter() const { return m_contentFilter.get(); }
    SCContentSharingSession* sharingSession() const { return m_sharingSession.get(); }
    WeakPtr<ScreenCaptureSessionSourceObserver> observer() const { return m_observer; }

    void updateContentFilter(SCContentFilter*);
    void streamDidEnd();

    bool operator==(const ScreenCaptureSessionSource&) const;

private:
    ScreenCaptureSessionSource(WeakPtr<ScreenCaptureSessionSourceObserver>&&, RetainPtr<SCStream>&&, RetainPtr<SCContentFilter>&&, RetainPtr<SCContentSharingSession>&&, CleanupFunction&&);

    RetainPtr<SCStream> m_stream;
    RetainPtr<SCContentFilter> m_contentFilter;
    RetainPtr<SCContentSharingSession> m_sharingSession;
    WeakPtr<ScreenCaptureSessionSourceObserver> m_observer;
    CleanupFunction m_cleanupFunction;
};

class ScreenCaptureKitSharingSessionManager : public CanMakeWeakPtr<ScreenCaptureKitSharingSessionManager> {
public:
    WEBCORE_EXPORT static ScreenCaptureKitSharingSessionManager& singleton();
    WEBCORE_EXPORT static bool isAvailable();
    WEBCORE_EXPORT static bool useSCContentSharingPicker();

    ScreenCaptureKitSharingSessionManager();
    ~ScreenCaptureKitSharingSessionManager();

    void sharingSessionDidChangeContent(SCContentSharingSession*);
    void sharingSessionDidEnd(SCContentSharingSession*);
    void contentSharingPickerSelectedFilterForStream(SCContentFilter*, SCStream*);
    void contentSharingPickerFailedWithError(NSError*);
    void contentSharingPickerUpdatedFilterForStream(SCContentFilter*, SCStream*);

    void cancelPicking();

    std::pair<RetainPtr<SCContentFilter>, RetainPtr<SCContentSharingSession>> contentFilterAndSharingSessionFromCaptureDevice(const CaptureDevice&);
    RefPtr<ScreenCaptureSessionSource> createSessionSourceForDevice(WeakPtr<ScreenCaptureSessionSourceObserver>, SCContentFilter*, SCContentSharingSession*, SCStreamConfiguration*, SCStreamDelegate*);
    void cancelPendingSessionForDevice(const CaptureDevice&);

    WEBCORE_EXPORT void promptForGetDisplayMedia(DisplayCapturePromptType, CompletionHandler<void(std::optional<CaptureDevice>)>&&);
    WEBCORE_EXPORT void cancelGetDisplayMediaPrompt();

    void cleanupSharingSession(SCContentSharingSession*);

private:
    void cleanupAllSessions();
    void completeDeviceSelection(SCContentFilter*, SCContentSharingSession* = nullptr);

    bool promptWithSCContentSharingSession(DisplayCapturePromptType);
    bool promptWithSCContentSharingPicker(DisplayCapturePromptType);

    bool promptingInProgress() const;

    void cleanupSessionSource(ScreenCaptureSessionSource&);

    WeakPtr<ScreenCaptureSessionSource> findActiveSource(SCContentSharingSession*);

    Vector<WeakPtr<ScreenCaptureSessionSource>> m_activeSources;

    RetainPtr<SCContentSharingSession> m_pendingSession;
    RetainPtr<SCContentFilter> m_pendingContentFilter;

    RetainPtr<WebDisplayMediaPromptHelper> m_promptHelper;
    CompletionHandler<void(std::optional<CaptureDevice>)> m_completionHandler;
    std::unique_ptr<RunLoop::Timer> m_promptWatchdogTimer;
};

} // namespace WebCore

#endif // HAVE(SCREEN_CAPTURE_KIT)
