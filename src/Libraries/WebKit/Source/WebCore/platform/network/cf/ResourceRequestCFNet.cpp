/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 30, 2024.
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
#include "ResourceRequestCFNet.h"

#include "HTTPHeaderNames.h"
#include "PublicSuffixStore.h"
#include "RegistrableDomain.h"
#include "ResourceLoadPriority.h"
#include "ResourceRequest.h"
#include <dlfcn.h>
#include <pal/spi/cf/CFNetworkSPI.h>
#include <wtf/TZoneMallocInlines.h>
#include <wtf/cf/TypeCastsCF.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(ResourceRequest);

// FIXME: Make this a NetworkingContext property.
#if PLATFORM(IOS_FAMILY)
bool ResourceRequest::s_httpPipeliningEnabled = true;
#else
bool ResourceRequest::s_httpPipeliningEnabled = false;
#endif

void ResourceRequest::updateFromDelegatePreservingOldProperties(const ResourceRequest& delegateProvidedRequest)
{
    // These are things we don't want willSendRequest delegate to mutate or reset.
    ResourceLoadPriority oldPriority = priority();
    RefPtr<FormData> oldHTTPBody = httpBody();
    bool isHiddenFromInspector = hiddenFromInspector();
    auto oldRequester = requester();
    auto oldInitiatorIdentifier = initiatorIdentifier();
    auto oldInspectorInitiatorNodeIdentifier = inspectorInitiatorNodeIdentifier();
    auto oldAppInitiatedValue = isAppInitiated();
    auto oldPrivacyProxyFailClosedForUnreachableNonMainHosts = privacyProxyFailClosedForUnreachableNonMainHosts();
    auto oldUseAdvancedPrivacyProtections = useAdvancedPrivacyProtections();

    *this = delegateProvidedRequest;

    setPriority(oldPriority);
    setHTTPBody(WTFMove(oldHTTPBody));
    setHiddenFromInspector(isHiddenFromInspector);
    setRequester(oldRequester);
    setInitiatorIdentifier(oldInitiatorIdentifier);
    if (oldInspectorInitiatorNodeIdentifier)
        setInspectorInitiatorNodeIdentifier(*oldInspectorInitiatorNodeIdentifier);
    setIsAppInitiated(oldAppInitiatedValue);
    setPrivacyProxyFailClosedForUnreachableNonMainHosts(oldPrivacyProxyFailClosedForUnreachableNonMainHosts);
    setUseAdvancedPrivacyProtections(oldUseAdvancedPrivacyProtections);
}

bool ResourceRequest::httpPipeliningEnabled()
{
    return s_httpPipeliningEnabled;
}

void ResourceRequest::setHTTPPipeliningEnabled(bool flag)
{
    s_httpPipeliningEnabled = flag;
}

// FIXME: It is confusing that this function both sets connection count and determines maximum request count at network layer. This can and should be done separately.
unsigned initializeMaximumHTTPConnectionCountPerHost()
{
    static const unsigned preferredConnectionCount = 6;
    static const unsigned unlimitedRequestCount = 10000;

    _CFNetworkHTTPConnectionCacheSetLimit(kHTTPLoadWidth, preferredConnectionCount);
    unsigned maximumHTTPConnectionCountPerHost = _CFNetworkHTTPConnectionCacheGetLimit(kHTTPLoadWidth);

    Boolean keyExistsAndHasValidFormat = false;
    Boolean prefValue = CFPreferencesGetAppBooleanValue(CFSTR("WebKitEnableHTTPPipelining"), kCFPreferencesCurrentApplication, &keyExistsAndHasValidFormat);
    if (keyExistsAndHasValidFormat)
        ResourceRequest::setHTTPPipeliningEnabled(prefValue);

    // Use WebCore scheduler when we can't use request priorities with CFNetwork.
    if (!ResourceRequest::resourcePrioritiesEnabled())
        return maximumHTTPConnectionCountPerHost;

    _CFNetworkHTTPConnectionCacheSetLimit(kHTTPPriorityNumLevels, resourceLoadPriorityCount);
    _CFNetworkHTTPConnectionCacheSetLimit(kHTTPMinimumFastLanePriority, toPlatformRequestPriority(ResourceLoadPriority::Medium));

    return unlimitedRequestCount;
}

#if PLATFORM(IOS_FAMILY)
void initializeHTTPConnectionSettingsOnStartup()
{
    // This need to be called from WebKitInitialize so the calls happen early enough, before any requests are made. <rdar://problem/9691871>
    // Desktop doesn't have early initialization so it is not clear how this should be done there. The CFNetwork SPI probably
    // needs to become more forgiving.
    // We can't read settings here as this is called too early for that. All values need to be constants.
    static const unsigned preferredConnectionCount = 6;
    static const unsigned fastLaneConnectionCount = 1;
    _CFNetworkHTTPConnectionCacheSetLimit(kHTTPLoadWidth, preferredConnectionCount);
    _CFNetworkHTTPConnectionCacheSetLimit(kHTTPPriorityNumLevels, resourceLoadPriorityCount);
    _CFNetworkHTTPConnectionCacheSetLimit(kHTTPMinimumFastLanePriority, toPlatformRequestPriority(ResourceLoadPriority::Medium));
    _CFNetworkHTTPConnectionCacheSetLimit(kHTTPNumFastLanes, fastLaneConnectionCount);
}
#endif

CFStringRef ResourceRequest::isUserInitiatedKey()
{
    static CFStringRef key = CFSTR("ResourceRequestIsUserInitiatedKey");
    return key;
}

} // namespace WebCore
