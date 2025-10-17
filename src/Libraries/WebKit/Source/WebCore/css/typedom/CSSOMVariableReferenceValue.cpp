/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 20, 2024.
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
#include "CSSOMVariableReferenceValue.h"

#include "CSSUnparsedValue.h"
#include "ExceptionOr.h"
#include <wtf/TZoneMallocInlines.h>
#include <wtf/text/StringBuilder.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(CSSOMVariableReferenceValue);

ExceptionOr<Ref<CSSOMVariableReferenceValue>> CSSOMVariableReferenceValue::create(String&& variable, RefPtr<CSSUnparsedValue>&& fallback)
{
    if (!variable.startsWith("--"_s))
        return Exception { ExceptionCode::TypeError, "Custom Variable Reference needs to have \"--\" prefix."_s };
    
    return adoptRef(*new CSSOMVariableReferenceValue(WTFMove(variable), WTFMove(fallback)));
}

ExceptionOr<void> CSSOMVariableReferenceValue::setVariable(String&& variable)
{
    if (!variable.startsWith("--"_s))
        return Exception { ExceptionCode::TypeError, "Custom Variable Reference needs to have \"--\" prefix."_s };
    
    m_variable = WTFMove(variable);
    return { };
}

String CSSOMVariableReferenceValue::toString() const
{
    StringBuilder builder;
    serialize(builder, { });
    return builder.toString();
}

void CSSOMVariableReferenceValue::serialize(StringBuilder& builder, OptionSet<SerializationArguments> arguments) const
{
    builder.append("var("_s, m_variable);
    if (m_fallback) {
        builder.append(',');
        m_fallback->serialize(builder, arguments);
    }
    builder.append(')');
}

} // namespace WebCore
