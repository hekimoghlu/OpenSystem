/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 1, 2022.
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
#import "config.h"
#import "CoreIPCPersonNameComponents.h"

#if PLATFORM(COCOA)

#import <wtf/RetainPtr.h>
#import <wtf/TZoneMallocInlines.h>

namespace WebKit {

WTF_MAKE_TZONE_ALLOCATED_IMPL(CoreIPCPersonNameComponents);

CoreIPCPersonNameComponents::CoreIPCPersonNameComponents(NSPersonNameComponents *components)
    : m_namePrefix(components.namePrefix)
    , m_givenName(components.givenName)
    , m_middleName(components.middleName)
    , m_familyName(components.familyName)
    , m_nickname(components.nickname)
{
    if (components.phoneticRepresentation)
        m_phoneticRepresentation = makeUnique<CoreIPCPersonNameComponents>(components.phoneticRepresentation);
}

RetainPtr<id> CoreIPCPersonNameComponents::toID() const
{
    auto components = adoptNS([NSPersonNameComponents new]);
    components.get().namePrefix = nsStringNilIfNull(m_namePrefix);
    components.get().givenName = nsStringNilIfNull(m_givenName);
    components.get().middleName = nsStringNilIfNull(m_middleName);
    components.get().familyName = nsStringNilIfNull(m_familyName);
    components.get().nickname = nsStringNilIfNull(m_nickname);
    if (m_phoneticRepresentation)
        components.get().phoneticRepresentation = m_phoneticRepresentation->toID().get();

    return components;
}

} // namespace WebKit

#endif // PLATFORM(COCOA)
