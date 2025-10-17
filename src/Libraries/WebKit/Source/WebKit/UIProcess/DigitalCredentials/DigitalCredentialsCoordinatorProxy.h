/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 5, 2022.
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

#if ENABLE(WEB_AUTHN)

#include "MessageReceiver.h"
#include <WebCore/CredentialRequestOptions.h>
#include <WebCore/DigitalCredentialRequestOptions.h>
#include <WebCore/FrameIdentifier.h>
#include <wtf/CompletionHandler.h>
#include <wtf/Forward.h>
#include <wtf/Noncopyable.h>

namespace WebCore {
struct ExceptionData;
struct DigitalCredentialRequestOptions;
}

namespace WebKit {
class WebPageProxy;
struct SharedPreferencesForWebProcess;
struct FrameInfoData;

// FIXME: We need to define a DigitalWalletResponse structure (https://webkit.org/b/278148)
// we are just initially handling exceptions as we build this out.
using DigitalRequestCompletionHandler = CompletionHandler<void(const WebCore::ExceptionData&)>;

class DigitalCredentialsCoordinatorProxy final : public IPC::MessageReceiver, public RefCounted<DigitalCredentialsCoordinatorProxy> {
    WTF_MAKE_FAST_ALLOCATED;
    WTF_MAKE_NONCOPYABLE(DigitalCredentialsCoordinatorProxy);
public:
    static Ref<DigitalCredentialsCoordinatorProxy> create(WebPageProxy&);
    ~DigitalCredentialsCoordinatorProxy();

    void ref() const final { RefCounted::ref(); }
    void deref() const final { RefCounted::deref(); }

    std::optional<SharedPreferencesForWebProcess> sharedPreferencesForWebProcess() const;

private:
    explicit DigitalCredentialsCoordinatorProxy(WebPageProxy&);

    // IPC::MessageReceiver.
    void didReceiveMessage(IPC::Connection&, IPC::Decoder&) final;

    // Receivers.
    void requestDigitalCredential(WebCore::FrameIdentifier, FrameInfoData&&, WebCore::DigitalCredentialRequestOptions&&, DigitalRequestCompletionHandler&&);
    void cancel(CompletionHandler<void()>&&);

    WeakPtr<WebPageProxy> m_page;
};

} // namespace WebKit

#endif // ENABLE(WEB_AUTHN)
