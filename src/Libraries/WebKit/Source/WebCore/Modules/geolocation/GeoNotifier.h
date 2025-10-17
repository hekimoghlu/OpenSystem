/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 6, 2025.
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

#include "PositionOptions.h"
#include "Timer.h"
#include <wtf/Forward.h>
#include <wtf/RefCounted.h>
#include <wtf/RefPtr.h>

namespace WebCore {

class Geolocation;
class GeolocationPosition;
class GeolocationPositionError;
class PositionCallback;
class PositionErrorCallback;

class GeoNotifier : public RefCounted<GeoNotifier> {
public:
    static Ref<GeoNotifier> create(Geolocation& geolocation, Ref<PositionCallback>&& positionCallback, RefPtr<PositionErrorCallback>&& positionErrorCallback, PositionOptions&& options)
    {
        return adoptRef(*new GeoNotifier(geolocation, WTFMove(positionCallback), WTFMove(positionErrorCallback), WTFMove(options)));
    }

    const PositionOptions& options() const { return m_options; }
    void setFatalError(RefPtr<GeolocationPositionError>&&);

    bool useCachedPosition() const { return m_useCachedPosition; }
    void setUseCachedPosition();

    void runSuccessCallback(GeolocationPosition*); // FIXME: This should take a reference.
    void runErrorCallback(GeolocationPositionError&);

    void startTimerIfNeeded();
    void stopTimer();
    void timerFired();
    bool hasZeroTimeout() const;

    Ref<PositionCallback> protectedSuccessCallback() { return m_successCallback; }

private:
    GeoNotifier(Geolocation&, Ref<PositionCallback>&&, RefPtr<PositionErrorCallback>&&, PositionOptions&&);

    Ref<Geolocation> m_geolocation;
    Ref<PositionCallback> m_successCallback;
    RefPtr<PositionErrorCallback> m_errorCallback;
    PositionOptions m_options;
    Timer m_timer;
    RefPtr<GeolocationPositionError> m_fatalError;
    bool m_useCachedPosition;
};

} // namespace WebCore

#endif // ENABLE(GEOLOCATION)
