/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 14, 2024.
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

#include "IDBKeyPath.h"
#include "IndexKey.h"
#include <wtf/Forward.h>

namespace JSC {
class CallFrame;
class JSGlobalObject;
class JSValue;
}

namespace WebCore {

class IDBIndexInfo;
class IDBKey;
class IDBKeyData;
class IDBObjectStoreInfo;
class IDBValue;
class IndexKey;
class JSDOMGlobalObject;

RefPtr<IDBKey> maybeCreateIDBKeyFromScriptValueAndKeyPath(JSC::JSGlobalObject&, JSC::JSValue, const IDBKeyPath&);
bool canInjectIDBKeyIntoScriptValue(JSC::JSGlobalObject&, JSC::JSValue, const IDBKeyPath&);
bool injectIDBKeyIntoScriptValue(JSC::JSGlobalObject&, const IDBKeyData&, JSC::JSValue, const IDBKeyPath&);

void generateIndexKeyForValue(JSC::JSGlobalObject&, const IDBIndexInfo&, JSC::JSValue, IndexKey& outKey, const std::optional<IDBKeyPath>&, const IDBKeyData&);

IndexIDToIndexKeyMap generateIndexKeyMapForValueIsolatedCopy(JSC::JSGlobalObject&, const IDBObjectStoreInfo&, const IDBKeyData&, const IDBValue&);

Ref<IDBKey> scriptValueToIDBKey(JSC::JSGlobalObject&, JSC::JSValue);

JSC::JSValue deserializeIDBValueToJSValue(JSC::JSGlobalObject&, const IDBValue&, Vector<std::pair<String, String>>&);
WEBCORE_EXPORT JSC::JSValue deserializeIDBValueToJSValue(JSC::JSGlobalObject&, const IDBValue&);
JSC::JSValue toJS(JSC::JSGlobalObject*, JSDOMGlobalObject*, const IDBValue&);
JSC::JSValue toJS(JSC::JSGlobalObject&, JSC::JSGlobalObject&, IDBKey*);
JSC::JSValue toJS(JSC::JSGlobalObject*, JSDOMGlobalObject*, const IDBKeyData&);

std::optional<JSC::JSValue> deserializeIDBValueWithKeyInjection(JSC::JSGlobalObject&, const IDBValue&, const IDBKeyData&, const std::optional<IDBKeyPath>&);

WEBCORE_EXPORT void callOnIDBSerializationThreadAndWait(Function<void(JSC::JSGlobalObject&)>&&);

}
