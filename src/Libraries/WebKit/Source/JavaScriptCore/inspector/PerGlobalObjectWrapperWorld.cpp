/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 10, 2024.
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
#include "PerGlobalObjectWrapperWorld.h"

namespace Inspector {

using namespace JSC;

JSValue PerGlobalObjectWrapperWorld::getWrapper(JSGlobalObject* globalObject)
{
    auto it = m_wrappers.find(globalObject);
    if (it != m_wrappers.end())
        return it->value.get();
    return JSValue();
}

void PerGlobalObjectWrapperWorld::addWrapper(JSGlobalObject* globalObject, JSObject* object)
{
    Strong<JSObject> wrapper(globalObject->vm(), object);
    m_wrappers.add(globalObject, wrapper);
}

void PerGlobalObjectWrapperWorld::clearAllWrappers()
{
    m_wrappers.clear();
}

} // namespace Inspector
