/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 6, 2023.
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

#include "FrameLoader.h"
#include "LocalFrameLoaderClient.h"
#include "ResourceRequest.h"
#include <wtf/WeakPtr.h>
#include <wtf/text/WTFString.h>

#if ENABLE(CONTENT_FILTERING)
#include "ContentFilterUnblockHandler.h"
#endif

namespace WebCore {
class PolicyChecker;
}

namespace WTF {
template<typename T> struct IsDeprecatedWeakRefSmartPointerException;
template<> struct IsDeprecatedWeakRefSmartPointerException<WebCore::PolicyChecker> : std::true_type { };
}

namespace WTF {
template<typename> class CompletionHandler;
class CompletionHandlerCallingScope;
}

namespace WebCore {

class DocumentLoader;
class FormState;
class HitTestResult;
class LocalFrame;
class NavigationAction;
class ResourceError;
class ResourceResponse;
class URLKeepingBlobAlive;

enum class NavigationPolicyDecision : uint8_t {
    ContinueLoad,
    IgnoreLoad,
    LoadWillContinueInAnotherProcess,
};

enum class PolicyDecisionMode { Synchronous, Asynchronous };

class PolicyChecker : public CanMakeWeakPtr<PolicyChecker> {
    WTF_MAKE_NONCOPYABLE(PolicyChecker);
    WTF_MAKE_FAST_ALLOCATED_WITH_HEAP_IDENTIFIER(Loader);
public:
    explicit PolicyChecker(LocalFrame&);

    using NavigationPolicyDecisionFunction = CompletionHandler<void(ResourceRequest&&, WeakPtr<FormState>&&, NavigationPolicyDecision)>;
    using NewWindowPolicyDecisionFunction = CompletionHandler<void(const ResourceRequest&, WeakPtr<FormState>&&, const AtomString& frameName, const NavigationAction&, ShouldContinuePolicyCheck)>;

    void checkNavigationPolicy(ResourceRequest&&, const ResourceResponse& redirectResponse, DocumentLoader*, RefPtr<FormState>&&, NavigationPolicyDecisionFunction&&, PolicyDecisionMode = PolicyDecisionMode::Asynchronous);
    void checkNavigationPolicy(ResourceRequest&&, const ResourceResponse& redirectResponse, NavigationPolicyDecisionFunction&&);
    void checkNewWindowPolicy(NavigationAction&&, ResourceRequest&&, RefPtr<FormState>&&, const AtomString& frameName, NewWindowPolicyDecisionFunction&&);

    void stopCheck();

    void cannotShowMIMEType(const ResourceResponse&);

    FrameLoadType loadType() const { return m_loadType; }
    void setLoadType(FrameLoadType loadType) { m_loadType = loadType; }

    bool delegateIsDecidingNavigationPolicy() const { return m_delegateIsDecidingNavigationPolicy; }
    bool delegateIsHandlingUnimplementablePolicy() const { return m_delegateIsHandlingUnimplementablePolicy; }

#if ENABLE(CONTENT_FILTERING)
    void setContentFilterUnblockHandler(ContentFilterUnblockHandler unblockHandler) { m_contentFilterUnblockHandler = WTFMove(unblockHandler); }
#endif

private:
    void handleUnimplementablePolicy(const ResourceError&);
    URLKeepingBlobAlive extendBlobURLLifetimeIfNecessary(const ResourceRequest&, const Document&, PolicyDecisionMode = PolicyDecisionMode::Asynchronous) const;
    std::optional<HitTestResult> hitTestResult(const NavigationAction&);

    Ref<LocalFrame> protectedFrame() const;

    WeakRef<LocalFrame> m_frame;

    uint64_t m_javaScriptURLPolicyCheckIdentifier { 0 };
    bool m_delegateIsDecidingNavigationPolicy;
    bool m_delegateIsHandlingUnimplementablePolicy;

    // This identifies the type of navigation action which prompted this load. Note 
    // that WebKit conveys this value as the WebActionNavigationTypeKey value
    // on navigation action delegate callbacks.
    FrameLoadType m_loadType;

#if ENABLE(CONTENT_FILTERING)
    ContentFilterUnblockHandler m_contentFilterUnblockHandler;
#endif
};

} // namespace WebCore
