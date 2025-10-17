/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 8, 2024.
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

#include "CSSStyleValue.h"
#include <wtf/RefCounted.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

template<typename> class ExceptionOr;
class CSSUnparsedValue;

class CSSOMVariableReferenceValue : public RefCounted<CSSOMVariableReferenceValue> {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(CSSOMVariableReferenceValue);
public:
    static ExceptionOr<Ref<CSSOMVariableReferenceValue>> create(String&&, RefPtr<CSSUnparsedValue>&& fallback = { });
    
    ExceptionOr<void> setVariable(String&&);
    String toString() const;
    void serialize(StringBuilder&, OptionSet<SerializationArguments>) const;

    const String& variable() const { return m_variable; }
    CSSUnparsedValue* fallback() { return m_fallback.get(); }
    
private:
    CSSOMVariableReferenceValue(String&& variable, RefPtr<CSSUnparsedValue>&& fallback)
        : m_variable(WTFMove(variable))
        , m_fallback(WTFMove(fallback)) { }
    
    String m_variable;
    RefPtr<CSSUnparsedValue> m_fallback;
};

} // namespace WebCore
