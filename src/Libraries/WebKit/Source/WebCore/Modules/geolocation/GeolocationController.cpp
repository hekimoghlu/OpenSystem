/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 10, 2022.
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
#include "GeolocationController.h"

#if ENABLE(GEOLOCATION)

#include "GeolocationClient.h"
#include "GeolocationError.h"
#include "GeolocationPositionData.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(GeolocationController);

GeolocationController::GeolocationController(Page& page, GeolocationClient& client)
    : m_page(page)
    , m_client(client)
{
    page.addActivityStateChangeObserver(*this);
}

GeolocationController::~GeolocationController()
{
    ASSERT(m_observers.isEmpty());

    // NOTE: We don't have to remove ourselves from page's ActivityStateChangeObserver set, since
    // we are supplement of the Page, and our destructor getting called means the page is being
    // torn down.

    auto* client = std::exchange(m_client, nullptr).get();
    client->geolocationDestroyed(); // This will destroy the client.
}

void GeolocationController::didNavigatePage()
{
    while (!m_observers.isEmpty())
        removeObserver(m_observers.begin()->get());
}

GeolocationClient& GeolocationController::client()
{
    return *m_client;
}

void GeolocationController::addObserver(Geolocation& observer, bool enableHighAccuracy)
{
    bool highAccuracyWasRequired = needsHighAccuracy();

    m_observers.add(observer);
    if (enableHighAccuracy)
        m_highAccuracyObservers.add(observer);

    if (m_isUpdating) {
        if (!highAccuracyWasRequired && enableHighAccuracy)
            m_client->setEnableHighAccuracy(true);
    } else
        startUpdatingIfNecessary();
}

void GeolocationController::removeObserver(Geolocation& observer)
{
    if (!m_observers.contains(observer))
        return;

    bool highAccuracyWasRequired = needsHighAccuracy();

    m_observers.remove(observer);
    m_highAccuracyObservers.remove(observer);

    if (!m_isUpdating)
        return;

    if (m_observers.isEmpty())
        stopUpdatingIfNecessary();
    else if (highAccuracyWasRequired && !needsHighAccuracy())
        m_client->setEnableHighAccuracy(false);
}

void GeolocationController::revokeAuthorizationToken(const String& authorizationToken)
{
    m_client->revokeAuthorizationToken(authorizationToken);
}

void GeolocationController::requestPermission(Geolocation& geolocation)
{
    if (!m_page->isVisible()) {
        m_pendingPermissionRequest.add(geolocation);
        return;
    }

    m_client->requestPermission(geolocation);
}

void GeolocationController::cancelPermissionRequest(Geolocation& geolocation)
{
    if (m_pendingPermissionRequest.remove(geolocation))
        return;

    m_client->cancelPermissionRequest(geolocation);
}

void GeolocationController::positionChanged(const std::optional<GeolocationPositionData>& position)
{
    m_lastPosition = position;
    for (auto& observer : copyToVectorOf<Ref<Geolocation>>(m_observers))
        observer->positionChanged();
}

void GeolocationController::errorOccurred(GeolocationError& error)
{
    for (auto& observer : copyToVectorOf<Ref<Geolocation>>(m_observers))
        observer->setError(error);
}

std::optional<GeolocationPositionData> GeolocationController::lastPosition()
{
    if (m_lastPosition)
        return m_lastPosition.value();

    return m_client->lastPosition();
}

void GeolocationController::activityStateDidChange(OptionSet<ActivityState> oldActivityState, OptionSet<ActivityState> newActivityState)
{
    // Toggle GPS based on page visibility to save battery.
    auto changed = oldActivityState ^ newActivityState;
    if (changed & ActivityState::IsVisible && !m_observers.isEmpty()) {
        if (newActivityState & ActivityState::IsVisible)
            startUpdatingIfNecessary();
        else
            stopUpdatingIfNecessary();
    }

    if (!m_page->isVisible())
        return;

    auto pendedPermissionRequests = WTFMove(m_pendingPermissionRequest);
    for (auto& permissionRequest : pendedPermissionRequests)
        m_client->requestPermission(permissionRequest.get());
}

void GeolocationController::startUpdatingIfNecessary()
{
    if (m_isUpdating || !m_page->isVisible() || m_observers.isEmpty())
        return;

    m_client->startUpdating((*m_observers.random())->authorizationToken(), needsHighAccuracy());
    m_isUpdating = true;
}

void GeolocationController::stopUpdatingIfNecessary()
{
    if (!m_isUpdating)
        return;

    m_client->stopUpdating();
    m_isUpdating = false;
}

ASCIILiteral GeolocationController::supplementName()
{
    return "GeolocationController"_s;
}

void provideGeolocationTo(Page* page, GeolocationClient& client)
{
    ASSERT(page);
    Supplement<Page>::provideTo(page, GeolocationController::supplementName(), makeUnique<GeolocationController>(*page, client));
}
    
} // namespace WebCore

#endif // ENABLE(GEOLOCATION)
