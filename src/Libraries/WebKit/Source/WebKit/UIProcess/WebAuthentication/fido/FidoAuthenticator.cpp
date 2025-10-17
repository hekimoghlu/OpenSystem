/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 10, 2024.
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
#include "FidoAuthenticator.h"
#include <WebCore/WebAuthenticationUtils.h>

#if ENABLE(WEB_AUTHN)

#include "CtapDriver.h"

namespace WebKit {

FidoAuthenticator::FidoAuthenticator(Ref<CtapDriver>&& driver)
    : m_driver(WTFMove(driver))
{
    ASSERT(m_driver);
}

FidoAuthenticator::~FidoAuthenticator()
{
    if (RefPtr driver = m_driver)
        driver->cancel();
}

CtapDriver& FidoAuthenticator::driver() const
{
    ASSERT(m_driver);
    return *m_driver;
}

Ref<CtapDriver> FidoAuthenticator::releaseDriver()
{
    ASSERT(m_driver);
    return m_driver.releaseNonNull();
}

String FidoAuthenticator::transportForDebugging() const
{
    return WebCore::toString(driver().transport());
}

} // namespace WebKit

#endif // ENABLE(WEB_AUTHN)
