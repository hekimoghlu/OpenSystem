/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 12, 2022.
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

#include "JITOperations.h"
#include "StringObject.h"

namespace JSC {

class ObjectPrototype;
class RegExp;
class RegExpObject;

class StringPrototype final : public StringObject {
public:
    using Base = StringObject;
    // We explicitly exclude InterceptsGetOwnPropertySlotByIndexEvenWhenLengthIsNotZero because StringPrototype's wrapped JSString is always empty, thus, we do not need to
    // trap via getOwnPropertySlotByIndex.
    static constexpr unsigned StructureFlags = (Base::StructureFlags | HasStaticPropertyTable) & ~InterceptsGetOwnPropertySlotByIndexEvenWhenLengthIsNotZero;

    static StringPrototype* create(VM&, JSGlobalObject*, Structure*);

    inline static Structure* createStructure(VM&, JSGlobalObject*, JSValue);

    using JSObject::getOwnPropertySlotByIndex;

    DECLARE_INFO;

private:
    StringPrototype(VM&, Structure*);
    void finishCreation(VM&, JSGlobalObject*);
};
STATIC_ASSERT_ISO_SUBSPACE_SHARABLE(StringPrototype, StringObject);

JSC_DECLARE_JIT_OPERATION(operationStringProtoFuncReplaceGeneric, JSCell*, (JSGlobalObject*, EncodedJSValue thisValue, EncodedJSValue searchValue, EncodedJSValue replaceValue));
JSC_DECLARE_JIT_OPERATION(operationStringProtoFuncReplaceRegExpEmptyStr, JSCell*, (JSGlobalObject*, JSString* thisValue, RegExpObject* searchValue));
JSC_DECLARE_JIT_OPERATION(operationStringProtoFuncReplaceRegExpString, JSCell*, (JSGlobalObject*, JSString* thisValue, RegExpObject* searchValue, JSString* replaceValue));

void substituteBackreferences(StringBuilder& result, const String& replacement, StringView source, const int* ovector, RegExp*);
void substituteBackreferencesSlow(StringBuilder& result, StringView replacement, StringView source, const int* ovector, RegExp*, size_t firstDollarSignPosition);

JSC_DECLARE_HOST_FUNCTION(stringProtoFuncRepeatCharacter);
JSC_DECLARE_HOST_FUNCTION(stringProtoFuncSplitFast);
JSC_DECLARE_HOST_FUNCTION(stringProtoFuncSubstring);

JSC_DECLARE_HOST_FUNCTION(builtinStringIncludesInternal);
JSC_DECLARE_HOST_FUNCTION(builtinStringIndexOfInternal);

} // namespace JSC
