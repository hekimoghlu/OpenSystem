/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 14, 2022.
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

#include "ExceptionOr.h"
#include "ScriptWrappable.h"
#include <wtf/RefCounted.h>
#include <wtf/RefPtr.h>

namespace JSC {
class CallFrame;
class JSGlobalObject;
class JSValue;
}

namespace WebCore {

class IDBKey;
class ScriptExecutionContext;

class IDBKeyRange final : public ScriptWrappable, public RefCounted<IDBKeyRange> {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(IDBKeyRange);
public:
    static Ref<IDBKeyRange> create(RefPtr<IDBKey>&& lower, RefPtr<IDBKey>&& upper, bool isLowerOpen, bool isUpperOpen);
    static Ref<IDBKeyRange> create(RefPtr<IDBKey>&&);
    ~IDBKeyRange();

    IDBKey* lower() const { return m_lower.get(); }
    IDBKey* upper() const { return m_upper.get(); }
    bool lowerOpen() const { return m_isLowerOpen; }
    bool upperOpen() const { return m_isUpperOpen; }

    static ExceptionOr<Ref<IDBKeyRange>> only(RefPtr<IDBKey>&& value);
    static ExceptionOr<Ref<IDBKeyRange>> only(JSC::JSGlobalObject&, JSC::JSValue key);

    static ExceptionOr<Ref<IDBKeyRange>> lowerBound(JSC::JSGlobalObject&, JSC::JSValue bound, bool open);
    static ExceptionOr<Ref<IDBKeyRange>> upperBound(JSC::JSGlobalObject&, JSC::JSValue bound, bool open);

    static ExceptionOr<Ref<IDBKeyRange>> bound(JSC::JSGlobalObject&, JSC::JSValue lower, JSC::JSValue upper, bool lowerOpen, bool upperOpen);

    ExceptionOr<bool> includes(JSC::JSGlobalObject&, JSC::JSValue key);

    WEBCORE_EXPORT bool isOnlyKey() const;

private:
    IDBKeyRange(RefPtr<IDBKey>&& lower, RefPtr<IDBKey>&& upper, bool isLowerOpen, bool isUpperOpen);

    RefPtr<IDBKey> m_lower;
    RefPtr<IDBKey> m_upper;
    bool m_isLowerOpen;
    bool m_isUpperOpen;
};

} // namespace WebCore
