/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 14, 2023.
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
#include "UserMessageHandlerDescriptor.h"

#if ENABLE(USER_MESSAGE_HANDLERS)

#include "DOMWrapperWorld.h"

namespace WebCore {

UserMessageHandlerDescriptor::UserMessageHandlerDescriptor(const AtomString& name, DOMWrapperWorld& world)
    : m_name(name)
    , m_world(world)
{
}

UserMessageHandlerDescriptor::~UserMessageHandlerDescriptor() = default;

const AtomString& UserMessageHandlerDescriptor::name() const
{
    return m_name;
}

DOMWrapperWorld& UserMessageHandlerDescriptor::world()
{
    return m_world.get();
}

const DOMWrapperWorld& UserMessageHandlerDescriptor::world() const
{
    return m_world.get();
}

} // namespace WebCore

#endif // ENABLE(USER_MESSAGE_HANDLERS)
