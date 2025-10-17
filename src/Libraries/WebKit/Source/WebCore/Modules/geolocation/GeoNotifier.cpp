/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 16, 2022.
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
#include "config.h"
#include "GeoNotifier.h"

#if ENABLE(GEOLOCATION)

#include "Geolocation.h"

namespace WebCore {

GeoNotifier::GeoNotifier(Geolocation& geolocation, Ref<PositionCallback>&& successCallback, RefPtr<PositionErrorCallback>&& errorCallback, PositionOptions&& options)
    : m_geolocation(geolocation)
    , m_successCallback(WTFMove(successCallback))
    , m_errorCallback(WTFMove(errorCallback))
    , m_options(WTFMove(options))
    , m_timer(*this, &GeoNotifier::timerFired)
    , m_useCachedPosition(false)
{
}

void GeoNotifier::setFatalError(RefPtr<GeolocationPositionError>&& error)
{
    // If a fatal error has already been set, stick with it. This makes sure that
    // when permission is denied, this is the error reported, as required by the
    // spec.
    if (m_fatalError)
        return;

    m_fatalError = WTFMove(error);
    // An existing timer may not have a zero timeout.
    m_timer.stop();
    m_timer.startOneShot(0_s);
}

void GeoNotifier::setUseCachedPosition()
{
    m_useCachedPosition = true;
    m_timer.startOneShot(0_s);
}

bool GeoNotifier::hasZeroTimeout() const
{
    return !m_options.timeout;
}

void GeoNotifier::runSuccessCallback(GeolocationPosition* position)
{
    // If we are here and the Geolocation permission is not approved, something has
    // gone horribly wrong.
    if (!m_geolocation->isAllowed())
        CRASH();

    protectedSuccessCallback()->handleEvent(position);
}

void GeoNotifier::runErrorCallback(GeolocationPositionError& error)
{
    if (RefPtr errorCallback = m_errorCallback)
        errorCallback->handleEvent(error);
}

void GeoNotifier::startTimerIfNeeded()
{
    m_timer.startOneShot(1_ms * m_options.timeout);
}

void GeoNotifier::stopTimer()
{
    m_timer.stop();
}

void GeoNotifier::timerFired()
{
    m_timer.stop();

    // Test for fatal error first. This is required for the case where the Frame is
    // disconnected and requests are cancelled.
    Ref geolocation = m_geolocation;
    if (RefPtr fatalError = m_fatalError) {
        runErrorCallback(*fatalError);
        // This will cause this notifier to be deleted.
        geolocation->fatalErrorOccurred(this);
        return;
    }

    if (m_useCachedPosition) {
        // Clear the cached position flag in case this is a watch request, which
        // will continue to run.
        m_useCachedPosition = false;
        geolocation->requestUsesCachedPosition(this);
        return;
    }
    
    if (m_errorCallback) {
        auto error = GeolocationPositionError::create(GeolocationPositionError::TIMEOUT, "Timeout expired"_s);
        m_errorCallback->handleEvent(error);
    }
    geolocation->requestTimedOut(this);
}

} // namespace WebCore

#endif // ENABLE(GEOLOCATION)
