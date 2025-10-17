/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 8, 2025.
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
#include "FloatingPointEnvironment.h"

#include <wtf/MainThread.h>
#include <wtf/NeverDestroyed.h>

namespace WebCore {

FloatingPointEnvironment& FloatingPointEnvironment::singleton()
{
    static NeverDestroyed<FloatingPointEnvironment> floatingPointEnvironment;
    return floatingPointEnvironment;
}

#if PLATFORM(IOS_FAMILY) && (CPU(ARM) || CPU(ARM64))

FloatingPointEnvironment::FloatingPointEnvironment()
    : m_isInitialized(false)
{
}

void FloatingPointEnvironment::enableDenormalSupport()
{
    RELEASE_ASSERT(isUIThread());
#if defined _ARM_ARCH_7
    fenv_t env;
    fegetenv(&env); 
    env.__fpscr &= ~0x01000000U;
    fesetenv(&env); 
#endif
    // Supporting denormal mode is already the default on x86, x86_64, and ARM64.
}

void FloatingPointEnvironment::saveMainThreadEnvironment()
{
    RELEASE_ASSERT(!m_isInitialized);
    RELEASE_ASSERT(isUIThread());
    fegetenv(&m_mainThreadEnvironment);
    m_isInitialized = true;
}

void FloatingPointEnvironment::propagateMainThreadEnvironment()
{
    RELEASE_ASSERT(m_isInitialized);
    RELEASE_ASSERT(!isUIThread());
    fesetenv(&m_mainThreadEnvironment);
}

#endif // PLATFORM(IOS_FAMILY) && (CPU(ARM) || CPU(ARM64))

} // namespace WebCore
