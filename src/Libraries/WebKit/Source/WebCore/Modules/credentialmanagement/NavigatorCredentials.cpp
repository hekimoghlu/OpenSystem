/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 14, 2022.
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
#include "NavigatorCredentials.h"

#if ENABLE(WEB_AUTHN)

#include "Document.h"
#include "LocalFrame.h"
#include "Navigator.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(NavigatorCredentials);

NavigatorCredentials::NavigatorCredentials() = default;

NavigatorCredentials::~NavigatorCredentials() = default;

ASCIILiteral NavigatorCredentials::supplementName()
{
    return "NavigatorCredentials"_s;
}

CredentialsContainer* NavigatorCredentials::credentials(WeakPtr<Document, WeakPtrImplWithEventTargetData>&& document)
{
    if (!m_credentialsContainer)
        m_credentialsContainer = CredentialsContainer::create(WTFMove(document));

    return m_credentialsContainer.get();
}

CredentialsContainer* NavigatorCredentials::credentials(Navigator& navigator)
{
    if (!navigator.frame() || !navigator.frame()->document())
        return nullptr;
    return NavigatorCredentials::from(&navigator)->credentials(*navigator.frame()->document());
}

NavigatorCredentials* NavigatorCredentials::from(Navigator* navigator)
{
    NavigatorCredentials* supplement = static_cast<NavigatorCredentials*>(Supplement<Navigator>::from(navigator, supplementName()));
    if (!supplement) {
        auto newSupplement = makeUnique<NavigatorCredentials>();
        supplement = newSupplement.get();
        provideTo(navigator, supplementName(), WTFMove(newSupplement));
    }
    return supplement;
}

} // namespace WebCore

#endif // ENABLE(WEB_AUTHN)
