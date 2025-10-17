/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 21, 2025.
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
#include "NavigationDestination.h"

#include "JSDOMGlobalObject.h"
#include "SerializedScriptValue.h"
#include <JavaScriptCore/JSCJSValueInlines.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(NavigationDestination);

NavigationDestination::NavigationDestination(const URL& url, RefPtr<NavigationHistoryEntry>&& entry, bool isSameDocument)
    : m_entry(WTFMove(entry))
    , m_url(url)
    , m_isSameDocument(isSameDocument)
{
}

JSC::JSValue NavigationDestination::getState(JSDOMGlobalObject& globalObject) const
{
    if (!m_stateObject)
        return JSC::jsUndefined();

    return m_stateObject->deserialize(globalObject, &globalObject, SerializationErrorMode::Throwing);
}

} // namespace WebCore
