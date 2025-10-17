/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 4, 2024.
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
#if !__has_feature(objc_arc)
#error This file requires ARC. Add the "-fobjc-arc" compiler flag for this file.
#endif

#import "config.h"
#import "WebExtensionContext.h"

#if ENABLE(WK_WEB_EXTENSIONS)

#import "CocoaHelpers.h"
#import "WKWebExtensionContext.h"
#import "WKWebExtensionControllerDelegatePrivate.h"
#import "WKWebExtensionControllerInternal.h"
#import "WKWebExtensionMatchPattern.h"
#import "WKWebExtensionMatchPatternInternal.h"
#import "WKWebExtensionPermission.h"
#import "WKWebViewInternal.h"
#import "WebExtensionContextProxy.h"
#import "WebExtensionContextProxyMessages.h"
#import "WebExtensionController.h"
#import "WebExtensionMatchPattern.h"
#import <wtf/BlockPtr.h>
#import <wtf/CallbackAggregator.h>

namespace WebKit {

void WebExtensionContext::permissionsGetAll(CompletionHandler<void(Vector<String>&&, Vector<String>&&)>&& completionHandler)
{
    Vector<String> permissions, origins;

    for (auto& permission : currentPermissions())
        permissions.append(permission);

    bool hasGrantedAccessToAllURLsOrHosts = false;

    for (auto& matchPattern : currentPermissionMatchPatterns()) {
        if (matchPattern->matchesAllHosts()) {
            hasGrantedAccessToAllURLsOrHosts = true;
            continue;
        }

        origins.append(matchPattern->string());
    }

    if (hasGrantedAccessToAllURLsOrHosts) {
        auto combinedPermissionMatchPatterns = protectedExtension()->combinedPermissionMatchPatterns();
        bool appendedMatchAllURLsOrHostsPattern = false;

        for (auto& matchPattern : combinedPermissionMatchPatterns) {
            if (matchPattern->matchesAllHosts()) {
                origins.append(matchPattern->string());
                appendedMatchAllURLsOrHostsPattern = true;
            }
        }

        // If we don't have the all URLs and hosts match pattern(s) in the manifest, access was requested implicitly (tabs, web navigation, etc.).
        if (!appendedMatchAllURLsOrHostsPattern)
            origins.append(WebExtensionMatchPattern::allHostsAndSchemesMatchPattern()->string());
    }

    completionHandler(WTFMove(permissions), WTFMove(origins));
}

void WebExtensionContext::permissionsContains(HashSet<String> permissions, HashSet<String> origins, CompletionHandler<void(bool)>&& completionHandler)
{
    completionHandler(hasPermissions(permissions, toPatterns(origins)));
}

void WebExtensionContext::permissionsRequest(HashSet<String> permissions, HashSet<String> origins, CompletionHandler<void(bool)>&& completionHandler)
{
    auto matchPatterns = toPatterns(origins);

    // If there is nothing to grant, return true. This matches Chrome and Firefox.
    if (permissions.isEmpty() && origins.isEmpty()) {
        firePermissionsEventListenerIfNecessary(WebExtensionEventListenerType::PermissionsOnAdded, permissions, matchPatterns);
        completionHandler(true);
        return;
    }

    if (hasPermissions(permissions, matchPatterns)) {
        completionHandler(true);
        return;
    }

    // There shouldn't be any unsupported permissions, since they got reported as an error before this.
    ASSERT(permissions.isSubset(extension().supportedPermissions()));

    bool requestedAllHostsPattern = WebExtensionMatchPattern::patternsMatchAllHosts(matchPatterns);
    if (requestedAllHostsPattern && !m_requestedOptionalAccessToAllHosts)
        m_requestedOptionalAccessToAllHosts = YES;

    class ResultHolder : public RefCounted<ResultHolder> {
    public:
        static Ref<ResultHolder> create() { return adoptRef(*new ResultHolder()); }
        ResultHolder() = default;

        bool matchPatternsAreGranted { false };
        bool permissionsAreGranted { false };

        MatchPatternSet neededMatchPatterns;
        PermissionsSet neededPermissions;

        WallTime matchPatternExpirationDate;
        WallTime permissionExpirationDate;
    };

    Ref resultHolder = ResultHolder::create();
    Ref callbackAggregator = CallbackAggregator::create([this, protectedThis = Ref { *this }, completionHandler = WTFMove(completionHandler), resultHolder, permissions, matchPatterns]() mutable {
        if (!resultHolder->matchPatternsAreGranted || !resultHolder->permissionsAreGranted) {
            completionHandler(false);
            return;
        }

        grantPermissionMatchPatterns(WTFMove(resultHolder->neededMatchPatterns), resultHolder->matchPatternExpirationDate);
        grantPermissions(WTFMove(resultHolder->neededPermissions), resultHolder->permissionExpirationDate);

        completionHandler(true);
    });

    requestPermissionMatchPatterns(matchPatterns, nullptr, [callbackAggregator, resultHolder](MatchPatternSet&& neededMatchPatterns, MatchPatternSet&& allowedMatchPatterns, WallTime expirationDate) {
        // The permissions.request() API only allows granting all or none.
        resultHolder->matchPatternsAreGranted = neededMatchPatterns.size() == allowedMatchPatterns.size();
        resultHolder->neededMatchPatterns = WTFMove(neededMatchPatterns);
        resultHolder->permissionExpirationDate = expirationDate;
    }, GrantOnCompletion::No, PermissionStateOptions::IncludeOptionalPermissions);

    requestPermissions(permissions, nullptr, [callbackAggregator, resultHolder](PermissionsSet&& neededPermissions, PermissionsSet&& allowedPermissions, WallTime expirationDate) {
        // The permissions.request() API only allows granting all or none.
        resultHolder->permissionsAreGranted = neededPermissions.size() == allowedPermissions.size();
        resultHolder->neededPermissions = WTFMove(neededPermissions);
        resultHolder->matchPatternExpirationDate = expirationDate;
    }, GrantOnCompletion::No, PermissionStateOptions::IncludeOptionalPermissions);
}

void WebExtensionContext::permissionsRemove(HashSet<String> permissions, HashSet<String> origins, CompletionHandler<void(bool)>&& completionHandler)
{
    auto matchPatterns = toPatterns(origins);
    bool removingAllHostsPattern = WebExtensionMatchPattern::patternsMatchAllHosts(matchPatterns);
    if (removingAllHostsPattern && m_requestedOptionalAccessToAllHosts)
        m_requestedOptionalAccessToAllHosts = false;

    removeGrantedPermissions(permissions);
    removeGrantedPermissionMatchPatterns(matchPatterns, EqualityOnly::No);

    completionHandler(!hasPermissions(permissions, matchPatterns));
}

void WebExtensionContext::firePermissionsEventListenerIfNecessary(WebExtensionEventListenerType type, const PermissionsSet& permissions, const MatchPatternSet& matchPatterns)
{
    ASSERT(type == WebExtensionEventListenerType::PermissionsOnAdded || type == WebExtensionEventListenerType::PermissionsOnRemoved);

    HashSet<String> origins = toStrings(matchPatterns);

    wakeUpBackgroundContentIfNecessaryToFireEvents({ type }, [=, this, protectedThis = Ref { *this }] {
        sendToProcessesForEvent(type, Messages::WebExtensionContextProxy::DispatchPermissionsEvent(type, permissions, origins));
    });
}

} // namespace WebKit

#endif // ENABLE(WK_WEB_EXTENSIONS)
