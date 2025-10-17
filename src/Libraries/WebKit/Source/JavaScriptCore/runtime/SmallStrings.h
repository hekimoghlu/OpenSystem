/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 22, 2023.
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

#include "CollectionScope.h"
#include "TypeofType.h"
#include <wtf/Noncopyable.h>

WTF_ALLOW_UNSAFE_BUFFER_USAGE_BEGIN

#define JSC_COMMON_STRINGS_EACH_NAME(macro) \
    macro(default) \
    macro(boolean) \
    macro(false) \
    macro(function) \
    macro(number) \
    macro(null) \
    macro(object) \
    macro(undefined) \
    macro(string) \
    macro(symbol) \
    macro(bigint) \
    macro(true)

namespace WTF {
class StringImpl;
}

namespace JSC {

class VM;
class JSString;

static constexpr unsigned maxSingleCharacterString = 0xFF;

class SmallStrings {
    WTF_MAKE_NONCOPYABLE(SmallStrings);
public:
    SmallStrings();
    ~SmallStrings();

    JSString* emptyString()
    {
        return m_emptyString;
    }

    JSString* singleCharacterString(unsigned char character)
    {
WTF_ALLOW_UNSAFE_BUFFER_USAGE_BEGIN
        return m_singleCharacterStrings[character];
WTF_ALLOW_UNSAFE_BUFFER_USAGE_END
    }

    JS_EXPORT_PRIVATE Ref<AtomStringImpl> singleCharacterStringRep(unsigned char character);

    void setIsInitialized(bool isInitialized) { m_isInitialized = isInitialized; }

    JSString** singleCharacterStrings() { return &m_singleCharacterStrings[0]; }

    void initializeCommonStrings(VM&);
    template<typename Visitor> void visitStrongReferences(Visitor&);

#define JSC_COMMON_STRINGS_ACCESSOR_DEFINITION(name) \
    JSString* name##String() const                   \
    {                                                \
        return m_##name;                             \
    }
    JSC_COMMON_STRINGS_EACH_NAME(JSC_COMMON_STRINGS_ACCESSOR_DEFINITION)
#undef JSC_COMMON_STRINGS_ACCESSOR_DEFINITION
    
    JSString* typeString(TypeofType type) const
    {
        switch (type) {
        case TypeofType::Undefined:
            return undefinedString();
        case TypeofType::Boolean:
            return booleanString();
        case TypeofType::Number:
            return numberString();
        case TypeofType::String:
            return stringString();
        case TypeofType::Symbol:
            return symbolString();
        case TypeofType::Object:
            return objectString();
        case TypeofType::Function:
            return functionString();
        case TypeofType::BigInt:
            return bigintString();
        }
        
        RELEASE_ASSERT_NOT_REACHED();
        return nullptr;
    }

    JSString* objectStringStart() const { return m_objectStringStart; }
    JSString* objectNullString() const { return m_objectNullString; }
    JSString* objectUndefinedString() const { return m_objectUndefinedString; }
    JSString* objectObjectString() const { return m_objectObjectString; }
    JSString* objectArrayString() const { return m_objectArrayString; }
    JSString* objectFunctionString() const { return m_objectFunctionString; }
    JSString* objectArgumentsString() const { return m_objectArgumentsString; }
    JSString* objectDateString() const { return m_objectDateString; }
    JSString* objectRegExpString() const { return m_objectRegExpString; }
    JSString* objectErrorString() const { return m_objectErrorString; }
    JSString* objectBooleanString() const { return m_objectBooleanString; }
    JSString* objectNumberString() const { return m_objectNumberString; }
    JSString* objectStringString() const { return m_objectStringString; }
    JSString* boundPrefixString() const { return m_boundPrefixString; }
    JSString* notEqualString() const { return m_notEqualString; }
    JSString* timedOutString() const { return m_timedOutString; }
    JSString* okString() const { return m_okString; }
    JSString* sentinelString() const { return m_sentinelString; }

    bool needsToBeVisited(CollectionScope scope) const
    {
        if (scope == CollectionScope::Full)
            return true;
        return m_needsToBeVisited;
    }

private:
    static constexpr unsigned singleCharacterStringCount = maxSingleCharacterString + 1;

    void initialize(VM*, JSString*&, ASCIILiteral value);

    JSString* m_emptyString { nullptr };
#define JSC_COMMON_STRINGS_ATTRIBUTE_DECLARATION(name) JSString* m_##name { nullptr };
    JSC_COMMON_STRINGS_EACH_NAME(JSC_COMMON_STRINGS_ATTRIBUTE_DECLARATION)
#undef JSC_COMMON_STRINGS_ATTRIBUTE_DECLARATION
    JSString* m_objectStringStart { nullptr };
    JSString* m_objectNullString { nullptr };
    JSString* m_objectUndefinedString { nullptr };
    JSString* m_objectObjectString { nullptr };
    JSString* m_objectArrayString { nullptr };
    JSString* m_objectFunctionString { nullptr };
    JSString* m_objectArgumentsString { nullptr };
    JSString* m_objectDateString { nullptr };
    JSString* m_objectRegExpString { nullptr };
    JSString* m_objectErrorString { nullptr };
    JSString* m_objectBooleanString { nullptr };
    JSString* m_objectNumberString { nullptr };
    JSString* m_objectStringString { nullptr };

    JSString* m_boundPrefixString { nullptr };
    JSString* m_notEqualString { nullptr };
    JSString* m_timedOutString { nullptr };
    JSString* m_okString { nullptr };
    JSString* m_sentinelString { nullptr };
    JSString* m_singleCharacterStrings[singleCharacterStringCount] { nullptr };
    bool m_needsToBeVisited { true };
    bool m_isInitialized { false };
};

} // namespace JSC

WTF_ALLOW_UNSAFE_BUFFER_USAGE_END
