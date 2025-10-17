/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 17, 2024.
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
#include "CDMFactory.h"

#if ENABLE(ENCRYPTED_MEDIA)

#include "CDMProxy.h"

#if ENABLE(THUNDER)
#include "CDMThunder.h"
#endif

namespace WebCore {

void CDMFactory::platformRegisterFactories(Vector<CDMFactory*>& factories)
{
#if ENABLE(THUNDER)
    factories.append(&CDMFactoryThunder::singleton());
#else
    UNUSED_PARAM(factories);
#endif
}

Vector<CDMProxyFactory*> CDMProxyFactory::platformRegisterFactories()
{
    Vector<CDMProxyFactory*> factories;
#if ENABLE(THUNDER)
    factories.reserveInitialCapacity(1);
    factories.append(&CDMFactoryThunder::singleton());
#endif
    return factories;
}

} // namespace WebCore

#endif // ENABLE(ENCRYPTED_MEDIA)
