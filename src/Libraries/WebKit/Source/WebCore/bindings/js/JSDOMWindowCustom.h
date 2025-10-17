/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 12, 2024.
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

#include "JSDOMWindow.h"

namespace WebCore {

class DOMWindow;
class Frame;

JSDOMWindow* asJSDOMWindow(JSC::JSGlobalObject*);
const JSDOMWindow* asJSDOMWindow(const JSC::JSGlobalObject*);

enum class DOMWindowType : bool { Local, Remote };
bool jsDOMWindowGetOwnPropertySlotRestrictedAccess(JSDOMGlobalObject*, DOMWindow&, JSC::JSGlobalObject&, JSC::PropertyName, JSC::PropertySlot&, const String&);

enum class CrossOriginObject : bool { Window, Location };
template<CrossOriginObject> void addCrossOriginOwnPropertyNames(JSC::JSGlobalObject&, JSC::PropertyNameArray&);

bool handleCommonCrossOriginProperties(JSC::JSObject* thisObject, JSC::VM&, JSC::PropertyName, JSC::PropertySlot&);

JSDOMWindow& mainWorldGlobalObject(LocalFrame&);
JSDOMWindow* mainWorldGlobalObject(LocalFrame*);

inline JSDOMWindow* asJSDOMWindow(JSC::JSGlobalObject* globalObject)
{
    return JSC::jsCast<JSDOMWindow*>(globalObject);
}

inline const JSDOMWindow* asJSDOMWindow(const JSC::JSGlobalObject* globalObject)
{
    return static_cast<const JSDOMWindow*>(globalObject);
}

inline JSDOMWindow* mainWorldGlobalObject(LocalFrame* frame)
{
    return frame ? &mainWorldGlobalObject(*frame) : nullptr;
}

JSC_DECLARE_CUSTOM_GETTER(showModalDialogGetter);

} // namespace WebCore
