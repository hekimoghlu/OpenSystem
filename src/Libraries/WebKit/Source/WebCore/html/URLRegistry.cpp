/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 17, 2022.
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
#include "URLRegistry.h"

#include <wtf/Lock.h>
#include <wtf/NeverDestroyed.h>
#include <wtf/TZoneMallocInlines.h>
#include <wtf/Vector.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(URLRegistry);

static Lock allRegistriesLock;
static Vector<URLRegistry*>& allRegistries() WTF_REQUIRES_LOCK(allRegistriesLock)
{
    static NeverDestroyed<Vector<URLRegistry*>> list;
    return list;
}

void URLRegistry::forEach(const Function<void(URLRegistry&)>& apply)
{
    Vector<URLRegistry*> registries;
    {
        Locker locker { allRegistriesLock };
        registries = allRegistries();
    }
    for (auto* registry : registries)
        apply(*registry);
}

URLRegistry::URLRegistry()
{
    Locker locker { allRegistriesLock };
    allRegistries().append(this);
}

URLRegistry::~URLRegistry()
{
    RELEASE_ASSERT_NOT_REACHED(); // All our registries are singleton objects.
}

} // namespace WebCore
