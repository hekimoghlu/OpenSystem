/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 26, 2024.
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

#if ENABLE(APPLE_PAY)

#include "ApplePaySetupConfiguration.h"
#include "JSDOMPromiseDeferred.h"
#include <WebCore/ActiveDOMObject.h>
#include <WebCore/JSDOMPromiseDeferredForward.h>
#include <wtf/Forward.h>
#include <wtf/Ref.h>
#include <wtf/RefCounted.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

class ApplePaySetupFeature;
class Document;

class ApplePaySetup : public ActiveDOMObject, public RefCounted<ApplePaySetup> {
public:
    void ref() const final { RefCounted::ref(); }
    void deref() const final { RefCounted::deref(); }

    static Ref<ApplePaySetup> create(ScriptExecutionContext&, ApplePaySetupConfiguration&&);

    using SetupFeaturesPromise = DOMPromiseDeferred<IDLSequence<IDLInterface<ApplePaySetupFeature>>>;
    void getSetupFeatures(Document&, SetupFeaturesPromise&&);

    using BeginPromise = DOMPromiseDeferred<IDLBoolean>;
    void begin(Document&, Vector<Ref<ApplePaySetupFeature>>&&, BeginPromise&&);

private:
    ApplePaySetup(ScriptExecutionContext&, ApplePaySetupConfiguration&&);

    // ActiveDOMObject
    void stop() final;
    void suspend(ReasonForSuspension) final;

    ApplePaySetupConfiguration m_configuration;
    std::optional<SetupFeaturesPromise> m_setupFeaturesPromise;
    std::optional<BeginPromise> m_beginPromise;
    RefPtr<PendingActivity<ApplePaySetup>> m_pendingActivity;
};

} // namespace WebCore

#endif // ENABLE(APPLE_PAY)
