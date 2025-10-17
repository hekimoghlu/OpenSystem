/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 29, 2023.
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

#include "MessageReceiver.h"
#include "WebGeolocationPosition.h"
#include "WebProcessSupplement.h"
#include <WebCore/RegistrableDomain.h>
#include <wtf/CheckedRef.h>
#include <wtf/Forward.h>
#include <wtf/Noncopyable.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/WeakHashMap.h>
#include <wtf/WeakHashSet.h>

namespace WebCore {
class Geolocation;
class GeolocationPositionData;
}

namespace WebKit {

class WebProcess;
class WebPage;

class WebGeolocationManager : public WebProcessSupplement, public IPC::MessageReceiver {
    WTF_MAKE_TZONE_ALLOCATED(WebGeolocationManager);
    WTF_MAKE_NONCOPYABLE(WebGeolocationManager);
public:
    explicit WebGeolocationManager(WebProcess&);
    ~WebGeolocationManager();

    void ref() const final;
    void deref() const final;

    static ASCIILiteral supplementName();

    void registerWebPage(WebPage&, const String& authorizationToken, bool needsHighAccuracy);
    void unregisterWebPage(WebPage&);
    void setEnableHighAccuracyForPage(WebPage&, bool);

private:
    // IPC::MessageReceiver
    void didReceiveMessage(IPC::Connection&, IPC::Decoder&) override;

    void didChangePosition(const WebCore::RegistrableDomain&, const WebCore::GeolocationPositionData&);
    void didFailToDeterminePosition(const WebCore::RegistrableDomain&, const String& errorMessage);
#if PLATFORM(IOS_FAMILY)
    void resetPermissions(const WebCore::RegistrableDomain&);
#endif // PLATFORM(IOS_FAMILY)

    struct PageSets {
        WeakHashSet<WebPage> pageSet;
        WeakHashSet<WebPage> highAccuracyPageSet;
    };
    bool isUpdating(const PageSets&) const;
    bool isHighAccuracyEnabled(const PageSets&) const;

    CheckedRef<WebProcess> m_process;
    HashMap<WebCore::RegistrableDomain, PageSets> m_pageSets;
    WeakHashMap<WebPage, WebCore::RegistrableDomain> m_pageToRegistrableDomain;
};

} // namespace WebKit
