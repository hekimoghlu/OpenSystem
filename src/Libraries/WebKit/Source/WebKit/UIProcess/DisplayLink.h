/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 12, 2022.
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

#if HAVE(DISPLAY_LINK)

#include "DisplayLinkObserverID.h"
#include <WebCore/AnimationFrameRate.h>
#include <WebCore/DisplayUpdate.h>
#include <WebCore/PlatformScreen.h>
#include <wtf/CheckedPtr.h>
#include <wtf/HashMap.h>
#include <wtf/Lock.h>
#include <wtf/TZoneMalloc.h>

#if PLATFORM(MAC)
#include <CoreVideo/CVDisplayLink.h>
#endif

#if PLATFORM(GTK) || PLATFORM(WPE)
#include "DisplayVBlankMonitor.h"
#endif

#if USE(WPE_BACKEND_PLAYSTATION)
struct wpe_playstation_display;
#endif

namespace WebKit {

class DisplayLink {
    WTF_MAKE_TZONE_ALLOCATED(DisplayLink);
public:
    class Client : public CanMakeThreadSafeCheckedPtr<Client> {
        WTF_MAKE_TZONE_ALLOCATED(Client);
        WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(Client);
    friend class DisplayLink;
    public:
        virtual ~Client() = default;

    private:
        virtual void displayLinkFired(WebCore::PlatformDisplayID, WebCore::DisplayUpdate, bool wantsFullSpeedUpdates, bool anyObserverWantsCallback) = 0;
    };

    explicit DisplayLink(WebCore::PlatformDisplayID);
    ~DisplayLink();

    WebCore::PlatformDisplayID displayID() const { return m_displayID; }
    WebCore::FramesPerSecond nominalFramesPerSecond() const { return m_displayNominalFramesPerSecond; }

    void displayPropertiesChanged();

    void addObserver(Client&, DisplayLinkObserverID, WebCore::FramesPerSecond);
    void removeObserver(Client&, DisplayLinkObserverID);

    void removeClient(Client&);

    // FIXME: Maybe callers should just register a DisplayLinkObserverID with the appropriate fps.
    void incrementFullSpeedRequestClientCount(Client&);
    void decrementFullSpeedRequestClientCount(Client&);

    void setObserverPreferredFramesPerSecond(Client&, DisplayLinkObserverID, WebCore::FramesPerSecond);

#if PLATFORM(GTK) || PLATFORM(WPE)
    DisplayVBlankMonitor& vblankMonitor() const { return *m_vblankMonitor; }
#endif

private:
#if PLATFORM(MAC)
    static CVReturn displayLinkCallback(CVDisplayLinkRef, const CVTimeStamp*, const CVTimeStamp*, CVOptionFlags, CVOptionFlags*, void* data);
    static WebCore::FramesPerSecond nominalFramesPerSecondFromDisplayLink(CVDisplayLinkRef);
#endif
    void notifyObserversDisplayDidRefresh();

    void platformInitialize();
    void platformFinalize();
    bool platformIsRunning() const;
    void platformStart();
    void platformStop();

    bool removeInfoForClientIfUnused(Client&) WTF_REQUIRES_LOCK(m_clientsLock);

    struct ObserverInfo {
        DisplayLinkObserverID observerID;
        WebCore::FramesPerSecond preferredFramesPerSecond;
    };

    struct ClientInfo {
        unsigned fullSpeedUpdatesClientCount { 0 };
        Vector<ObserverInfo> observers;
    };

#if PLATFORM(MAC)
    CVDisplayLinkRef m_displayLink { nullptr };
#endif
#if PLATFORM(GTK) || PLATFORM(WPE)
    std::unique_ptr<DisplayVBlankMonitor> m_vblankMonitor;
#endif
#if USE(WPE_BACKEND_PLAYSTATION)
    struct wpe_playstation_display* m_display;
#endif
    Lock m_clientsLock;
    HashMap<CheckedRef<Client>, ClientInfo> m_clients WTF_GUARDED_BY_LOCK(m_clientsLock);
    const WebCore::PlatformDisplayID m_displayID;
    WebCore::FramesPerSecond m_displayNominalFramesPerSecond { WebCore::FullSpeedFramesPerSecond };
    WebCore::DisplayUpdate m_currentUpdate;
    unsigned m_fireCountWithoutObservers { 0 };
};

class DisplayLinkCollection {
public:
    DisplayLink& displayLinkForDisplay(WebCore::PlatformDisplayID);
    DisplayLink* existingDisplayLinkForDisplay(WebCore::PlatformDisplayID) const;

    std::optional<unsigned> nominalFramesPerSecondForDisplay(WebCore::PlatformDisplayID);
    void startDisplayLink(DisplayLink::Client&, DisplayLinkObserverID, WebCore::PlatformDisplayID, WebCore::FramesPerSecond preferredFramesPerSecond);
    void stopDisplayLink(DisplayLink::Client&, DisplayLinkObserverID, WebCore::PlatformDisplayID);
    void stopDisplayLinks(DisplayLink::Client&);
    void setDisplayLinkPreferredFramesPerSecond(DisplayLink::Client&, DisplayLinkObserverID, WebCore::PlatformDisplayID, WebCore::FramesPerSecond preferredFramesPerSecond);
    void setDisplayLinkForDisplayWantsFullSpeedUpdates(DisplayLink::Client&, WebCore::PlatformDisplayID, bool wantsFullSpeedUpdates);

private:
    void add(std::unique_ptr<DisplayLink>&&);

    Vector<std::unique_ptr<DisplayLink>> m_displayLinks;
};

} // namespace WebKit

#endif // HAVE(DISPLAY_LINK)
