/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 17, 2025.
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

#if ENABLE(GEOLOCATION)

#include "ActiveDOMObject.h"
#include "Document.h"
#include "GeolocationPosition.h"
#include "GeolocationPositionError.h"
#include "PositionCallback.h"
#include "PositionErrorCallback.h"
#include "PositionOptions.h"
#include "ScriptWrappable.h"
#include "Timer.h"
#include <wtf/CheckedRef.h>
#include <wtf/HashMap.h>
#include <wtf/HashSet.h>
#include <wtf/RefCountedAndCanMakeWeakPtr.h>

namespace WebCore {

class GeoNotifier;
class GeolocationError;
class LocalFrame;
class Navigator;
class Page;
class ScriptExecutionContext;
class SecurityOrigin;
struct PositionOptions;

class Geolocation final : public RefCountedAndCanMakeWeakPtr<Geolocation>, public ScriptWrappable, public ActiveDOMObject {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED_EXPORT(Geolocation, WEBCORE_EXPORT);
    friend class GeoNotifier;
public:
    void ref() const final { RefCounted::ref(); }
    void deref() const final { RefCounted::deref(); }

    static Ref<Geolocation> create(Navigator&);
    WEBCORE_EXPORT ~Geolocation();

    WEBCORE_EXPORT void resetAllGeolocationPermission();
    Document* document() const { return downcast<Document>(scriptExecutionContext()); }

    void getCurrentPosition(Ref<PositionCallback>&&, RefPtr<PositionErrorCallback>&&, PositionOptions&&);
    int watchPosition(Ref<PositionCallback>&&, RefPtr<PositionErrorCallback>&&, PositionOptions&&);
    void clearWatch(int watchID);

    WEBCORE_EXPORT void setIsAllowed(bool, const String& authorizationToken);
    const String& authorizationToken() const { return m_authorizationToken; }
    WEBCORE_EXPORT void resetIsAllowed();
    bool isAllowed() const { return m_allowGeolocation == Yes; }

    void positionChanged();
    void setError(GeolocationError&);
    bool shouldBlockGeolocationRequests();

    Navigator* navigator();
    WEBCORE_EXPORT LocalFrame* frame() const;

private:
    explicit Geolocation(Navigator&);

    GeolocationPosition* lastPosition();

    // ActiveDOMObject.
    void stop() override;
    void suspend(ReasonForSuspension) override;
    void resume() override;

    bool isDenied() const { return m_allowGeolocation == No; }

    Page* page() const;
    SecurityOrigin* securityOrigin() const;
    RefPtr<SecurityOrigin> protectedSecurityOrigin() const;

    typedef Vector<RefPtr<GeoNotifier>> GeoNotifierVector;
    typedef HashSet<RefPtr<GeoNotifier>> GeoNotifierSet;

    class Watchers {
    public:
        bool add(int id, RefPtr<GeoNotifier>&&);
        GeoNotifier* find(int id);
        void remove(int id);
        void remove(GeoNotifier*);
        bool contains(GeoNotifier*) const;
        void clear();
        bool isEmpty() const;
        void getNotifiersVector(GeoNotifierVector&) const;
    private:
        typedef HashMap<int, RefPtr<GeoNotifier>> IdToNotifierMap;
        typedef HashMap<RefPtr<GeoNotifier>, int> NotifierToIdMap;
        IdToNotifierMap m_idToNotifierMap;
        NotifierToIdMap m_notifierToIdMap;
    };

    bool hasListeners() const { return !m_oneShots.isEmpty() || !m_watchers.isEmpty(); }

    void sendError(GeoNotifierVector&, GeolocationPositionError&);
    void sendPosition(GeoNotifierVector&, GeolocationPosition&);

    static void extractNotifiersWithCachedPosition(GeoNotifierVector& notifiers, GeoNotifierVector* cached);
    static void copyToSet(const GeoNotifierVector&, GeoNotifierSet&);

    static void stopTimer(GeoNotifierVector&);
    void stopTimersForOneShots();
    void stopTimersForWatchers();
    void stopTimers();

    void cancelRequests(GeoNotifierVector&);
    void cancelAllRequests();

    void makeSuccessCallbacks(GeolocationPosition&);
    void handleError(GeolocationPositionError&);

    void requestPermission();
    void revokeAuthorizationTokenIfNecessary();

    bool startUpdating(GeoNotifier*);
    void stopUpdating();

    void handlePendingPermissionNotifiers();

    void startRequest(GeoNotifier*);

    void fatalErrorOccurred(GeoNotifier*);
    void requestTimedOut(GeoNotifier*);
    void requestUsesCachedPosition(GeoNotifier*);
    bool haveSuitableCachedPosition(const PositionOptions&);
    void makeCachedPositionCallbacks();

    void resumeTimerFired();

    WeakPtr<Navigator> m_navigator;
    GeoNotifierSet m_oneShots;
    Watchers m_watchers;
    GeoNotifierSet m_pendingForPermissionNotifiers;
    RefPtr<GeolocationPosition> m_lastPosition;

    enum { Unknown, InProgress, Yes, No } m_allowGeolocation { Unknown };
    String m_authorizationToken;
    bool m_isSuspended { false };
    bool m_resetOnResume { false };
    bool m_hasChangedPosition { false };
    RefPtr<GeolocationPositionError> m_errorWaitingForResume;
    Timer m_resumeTimer;
    GeoNotifierSet m_requestsAwaitingCachedPosition;
};
    
} // namespace WebCore

#endif // ENABLE(GEOLOCATION)
