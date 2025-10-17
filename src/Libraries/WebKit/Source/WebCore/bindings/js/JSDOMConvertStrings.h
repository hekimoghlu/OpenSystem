/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 4, 2025.
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

#include "IDLTypes.h"
#include "JSDOMConvertBase.h"
#include "StringAdaptors.h"
#include "TrustedType.h"
#include <JavaScriptCore/JSGlobalObject.h>

namespace WebCore {

class ScriptExecutionContext;

WEBCORE_EXPORT String identifierToString(JSC::JSGlobalObject&, const JSC::Identifier&);

WEBCORE_EXPORT String identifierToByteString(JSC::JSGlobalObject&, const JSC::Identifier&);
WEBCORE_EXPORT ConversionResult<IDLByteString> valueToByteString(JSC::JSGlobalObject&, JSC::JSValue);
WEBCORE_EXPORT ConversionResult<IDLAtomStringAdaptor<IDLByteString>> valueToByteAtomString(JSC::JSGlobalObject&, JSC::JSValue);

WEBCORE_EXPORT String identifierToUSVString(JSC::JSGlobalObject&, const JSC::Identifier&);
WEBCORE_EXPORT ConversionResult<IDLUSVString> valueToUSVString(JSC::JSGlobalObject&, JSC::JSValue);
WEBCORE_EXPORT ConversionResult<IDLAtomStringAdaptor<IDLUSVString>> valueToUSVAtomString(JSC::JSGlobalObject&, JSC::JSValue);

ConversionResult<IDLDOMString> trustedScriptCompliantString(JSC::JSGlobalObject&, JSC::JSValue, const String& sink);

inline AtomString propertyNameToString(JSC::PropertyName propertyName)
{
    ASSERT(!propertyName.isSymbol());
    return propertyName.uid() ? propertyName.uid() : propertyName.publicName();
}

inline AtomString propertyNameToAtomString(JSC::PropertyName propertyName)
{
    return AtomString(propertyName.uid() ? propertyName.uid() : propertyName.publicName());
}

// MARK: -
// MARK: String types

template<> struct Converter<IDLDOMString> : DefaultConverter<IDLDOMString> {
    using Result = ConversionResult<IDLDOMString>;

    static Result convert(JSC::JSGlobalObject& lexicalGlobalObject, JSC::JSValue value)
    {
        auto& vm = lexicalGlobalObject.vm();
        auto scope = DECLARE_THROW_SCOPE(vm);

        auto string = value.toWTFString(&lexicalGlobalObject);

        RETURN_IF_EXCEPTION(scope, Result::exception());

        return Result { WTFMove(string) };
    }
};

template<> struct JSConverter<IDLDOMString> {
    static constexpr bool needsState = true;
    static constexpr bool needsGlobalObject = false;

    static JSC::JSValue convert(JSC::JSGlobalObject& lexicalGlobalObject, const String& value)
    {
        return JSC::jsStringWithCache(JSC::getVM(&lexicalGlobalObject), value);
    }

    static JSC::JSValue convert(JSC::JSGlobalObject& lexicalGlobalObject, const UncachedString& value)
    {
        return JSC::jsString(JSC::getVM(&lexicalGlobalObject), value.string);
    }

    static JSC::JSValue convert(JSC::JSGlobalObject& lexicalGlobalObject, const OwnedString& value)
    {
        return JSC::jsOwnedString(JSC::getVM(&lexicalGlobalObject), value.string);
    }

    static JSC::JSValue convert(JSC::JSGlobalObject& lexicalGlobalObject, const URL& value)
    {
        return JSC::jsOwnedString(JSC::getVM(&lexicalGlobalObject), value.string());
    }
};

template<> struct Converter<IDLByteString> : DefaultConverter<IDLByteString> {
    using Result = ConversionResult<IDLByteString>;

    static Result convert(JSC::JSGlobalObject& lexicalGlobalObject, JSC::JSValue value)
    {
        return valueToByteString(lexicalGlobalObject, value);
    }
};

template<> struct JSConverter<IDLByteString> {
    static constexpr bool needsState = true;
    static constexpr bool needsGlobalObject = false;

    static JSC::JSValue convert(JSC::JSGlobalObject& lexicalGlobalObject, const String& value)
    {
        return JSC::jsStringWithCache(JSC::getVM(&lexicalGlobalObject), value);
    }

    static JSC::JSValue convert(JSC::JSGlobalObject& lexicalGlobalObject, const UncachedString& value)
    {
        return JSC::jsString(JSC::getVM(&lexicalGlobalObject), value.string);
    }

    static JSC::JSValue convert(JSC::JSGlobalObject& lexicalGlobalObject, const OwnedString& value)
    {
        return JSC::jsOwnedString(JSC::getVM(&lexicalGlobalObject), value.string);
    }

    static JSC::JSValue convert(JSC::JSGlobalObject& lexicalGlobalObject, const URL& value)
    {
        return JSC::jsOwnedString(JSC::getVM(&lexicalGlobalObject), value.string());
    }
};

template<> struct Converter<IDLUSVString> : DefaultConverter<IDLUSVString> {
    using Result = ConversionResult<IDLUSVString>;

    static Result convert(JSC::JSGlobalObject& lexicalGlobalObject, JSC::JSValue value)
    {
        return valueToUSVString(lexicalGlobalObject, value);
    }

    static JSC::JSValue convert(JSC::JSGlobalObject& lexicalGlobalObject, const URL& value)
    {
        return JSC::jsOwnedString(JSC::getVM(&lexicalGlobalObject), value.string());
    }
};

template<> struct JSConverter<IDLUSVString> {
    static constexpr bool needsState = true;
    static constexpr bool needsGlobalObject = false;

    static JSC::JSValue convert(JSC::JSGlobalObject& lexicalGlobalObject, const String& value)
    {
        return JSC::jsStringWithCache(JSC::getVM(&lexicalGlobalObject), value);
    }

    static JSC::JSValue convert(JSC::JSGlobalObject& lexicalGlobalObject, const UncachedString& value)
    {
        return JSC::jsString(JSC::getVM(&lexicalGlobalObject), value.string);
    }

    static JSC::JSValue convert(JSC::JSGlobalObject& lexicalGlobalObject, const OwnedString& value)
    {
        return JSC::jsOwnedString(JSC::getVM(&lexicalGlobalObject), value.string);
    }

    static JSC::JSValue convert(JSC::JSGlobalObject& lexicalGlobalObject, const URL& value)
    {
        return JSC::jsOwnedString(JSC::getVM(&lexicalGlobalObject), value.string());
    }
};

// MARK: -
// MARK: String type adaptors

template<typename TrustedStringAdaptorIDL>
struct TrustedStringAdaptorTraits;

template<typename T> struct TrustedStringAdaptorTraits<IDLStringContextTrustedScriptAdaptor<T>> {
    static constexpr auto trustedType = TrustedType::TrustedScript;
};

template<typename IDL>
struct TrustedStringConverter : DefaultConverter<IDL> {
    using Result = ConversionResult<IDL>;
    using Traits = TrustedStringAdaptorTraits<IDL>;

    static Result convert(JSC::JSGlobalObject& lexicalGlobalObject, JSC::JSValue value)
    {
        return Converter<IDL>::convert(lexicalGlobalObject, value, emptyString());
    }

    static Result convert(JSC::JSGlobalObject& lexicalGlobalObject, JSC::JSValue value, const String& sink)
    {
        return trustedScriptCompliantString(lexicalGlobalObject, value, sink);
    }
};

template<typename InnerStringType>
struct JSStringAdaptorConverter  {
    static constexpr bool needsState = true;
    static constexpr bool needsGlobalObject = false;

    static JSC::JSValue convert(JSC::JSGlobalObject& lexicalGlobalObject, const String& value)
    {
        return JSConverter<InnerStringType>::convert(lexicalGlobalObject, value);
    }
};

template<typename InnerStringType>
struct JSAtomStringAdaptorConverter  {
    static constexpr bool needsState = true;
    static constexpr bool needsGlobalObject = false;

    static JSC::JSValue convert(JSC::JSGlobalObject& lexicalGlobalObject, const AtomString& value)
    {
        return JSConverter<InnerStringType>::convert(lexicalGlobalObject, value);
    }

    static JSC::JSValue convert(JSC::JSGlobalObject& lexicalGlobalObject, const URL& value)
    {
        return JSConverter<InnerStringType>::convert(lexicalGlobalObject, value.string());
    }
};

template<typename T> struct Converter<IDLStringContextTrustedScriptAdaptor<T>>
    : TrustedStringConverter<IDLStringContextTrustedScriptAdaptor<T>> { };

template<typename T> struct JSConverter<IDLLegacyNullToEmptyStringAdaptor<T>> : JSStringAdaptorConverter<T> { };
template<typename T> struct JSConverter<IDLStringContextTrustedScriptAdaptor<T>> : JSStringAdaptorConverter<T> { };
template<typename T> struct JSConverter<IDLLegacyNullToEmptyAtomStringAdaptor<T>> : JSAtomStringAdaptorConverter<T> { };

template<typename IDL> struct Converter<IDLLegacyNullToEmptyStringAdaptor<IDL>> : DefaultConverter<IDLLegacyNullToEmptyStringAdaptor<IDL>> {
    using Result = ConversionResult<IDLLegacyNullToEmptyStringAdaptor<IDL>>;

    static Result convert(JSC::JSGlobalObject& lexicalGlobalObject, JSC::JSValue value)
    {
        if (value.isNull())
            return emptyString();
        return WebCore::convert<IDL>(lexicalGlobalObject, value);
    }
};

template<typename T> struct Converter<IDLLegacyNullToEmptyAtomStringAdaptor<T>> : DefaultConverter<IDLLegacyNullToEmptyAtomStringAdaptor<T>> {
    using Result = ConversionResult<IDLLegacyNullToEmptyAtomStringAdaptor<T>>;

    static Result convert(JSC::JSGlobalObject& lexicalGlobalObject, JSC::JSValue value)
    {
        if (value.isNull())
            return emptyAtom();
        return WebCore::convert<IDLAtomStringAdaptor<T>>(lexicalGlobalObject, value);
    }
};

template<typename T> struct Converter<IDLAtomStringAdaptor<T>> : DefaultConverter<IDLAtomStringAdaptor<T>> {
    using Result = ConversionResult<IDLAtomStringAdaptor<T>>;

    static Result convert(JSC::JSGlobalObject& lexicalGlobalObject, JSC::JSValue value)
    {
        static_assert(std::is_same<T, IDLDOMString>::value, "This adaptor is only supported for IDLDOMString at the moment.");

        auto& vm = lexicalGlobalObject.vm();
        auto scope = DECLARE_THROW_SCOPE(vm);

        auto string = value.toString(&lexicalGlobalObject)->toAtomString(&lexicalGlobalObject);

        RETURN_IF_EXCEPTION(scope, Result::exception());

        return Result { WTFMove(string) };
    }
};

template<> struct Converter<IDLAtomStringAdaptor<IDLUSVString>> : DefaultConverter<IDLAtomStringAdaptor<IDLUSVString>> {
    using Result = ConversionResult<IDLAtomStringAdaptor<IDLUSVString>>;

    static Result convert(JSC::JSGlobalObject& lexicalGlobalObject, JSC::JSValue value)
    {
        return valueToUSVAtomString(lexicalGlobalObject, value);
    }
};

template<> struct Converter<IDLAtomStringAdaptor<IDLByteString>> : DefaultConverter<IDLAtomStringAdaptor<IDLByteString>> {
    using Result = ConversionResult<IDLAtomStringAdaptor<IDLByteString>>;

    static Result convert(JSC::JSGlobalObject& lexicalGlobalObject, JSC::JSValue value)
    {
        return valueToByteAtomString(lexicalGlobalObject, value);
    }
};

template<typename IDL>  struct JSConverter<IDLAtomStringAdaptor<IDL>> {
    static constexpr bool needsState = true;
    static constexpr bool needsGlobalObject = false;

    static JSC::JSValue convert(JSC::JSGlobalObject& lexicalGlobalObject, const AtomString& value)
    {
        return toJS<IDL>(lexicalGlobalObject, value);
    }

    static JSC::JSValue convert(JSC::JSGlobalObject& lexicalGlobalObject, const String& value)
    {
        return toJS<IDL>(lexicalGlobalObject, value);
    }

    static JSC::JSValue convert(JSC::JSGlobalObject& lexicalGlobalObject, const URL& value)
    {
        return toJS<IDL>(lexicalGlobalObject, value.string());
    }
};

template<>  struct JSConverter<IDLAtomStringAdaptor<IDLUSVString>> {
    static constexpr bool needsState = true;
    static constexpr bool needsGlobalObject = false;

    static JSC::JSValue convert(JSC::JSGlobalObject& lexicalGlobalObject, const AtomString& value)
    {
        return toJS<IDLUSVString>(lexicalGlobalObject, value.string());
    }

    static JSC::JSValue convert(JSC::JSGlobalObject& lexicalGlobalObject, const String& value)
    {
        return toJS<IDLUSVString>(lexicalGlobalObject, value);
    }

    static JSC::JSValue convert(JSC::JSGlobalObject& lexicalGlobalObject, const URL& value)
    {
        return toJS<IDLUSVString>(lexicalGlobalObject, value.string());
    }
};

template<> struct JSConverter<IDLAtomStringAdaptor<IDLByteString>> {
    static constexpr bool needsState = true;
    static constexpr bool needsGlobalObject = false;

    static JSC::JSValue convert(JSC::JSGlobalObject& lexicalGlobalObject, const AtomString& value)
    {
        return toJS<IDLByteString>(lexicalGlobalObject, value.string());
    }
};

template<typename IDL> struct Converter<IDLRequiresExistingAtomStringAdaptor<IDL>> : DefaultConverter<IDLRequiresExistingAtomStringAdaptor<IDL>> {
    static AtomString convert(JSC::JSGlobalObject& lexicalGlobalObject, JSC::JSValue value)
    {
        static_assert(std::is_same<IDL, IDLDOMString>::value, "This adaptor is only supported for IDLDOMString at the moment.");

        return value.toString(&lexicalGlobalObject)->toExistingAtomString(&lexicalGlobalObject);
    }
};

template<typename IDL> struct JSConverter<IDLRequiresExistingAtomStringAdaptor<IDL>> {
    static constexpr bool needsState = true;
    static constexpr bool needsGlobalObject = false;

    static JSC::JSValue convert(JSC::JSGlobalObject& lexicalGlobalObject, const AtomString& value)
    {
        static_assert(std::is_same<IDL, IDLDOMString>::value, "This adaptor is only supported for IDLDOMString at the moment.");

        return toJS<IDL>(lexicalGlobalObject, value);
    }
};


template<> struct JSConverter<IDLNullable<IDLDOMString>> {
    static constexpr bool needsState = true;
    static constexpr bool needsGlobalObject = false;

    static JSC::JSValue convert(JSC::JSGlobalObject& lexicalGlobalObject, const String& value)
    {
        if (value.isNull())
            return JSC::jsNull();
        return toJS<IDLDOMString>(lexicalGlobalObject, value);
    }

    static JSC::JSValue convert(JSC::JSGlobalObject& lexicalGlobalObject, const UncachedString& value)
    {
        if (value.string.isNull())
            return JSC::jsNull();
        return toJS<IDLDOMString>(lexicalGlobalObject, value);
    }

    static JSC::JSValue convert(JSC::JSGlobalObject& lexicalGlobalObject, const OwnedString& value)
    {
        if (value.string.isNull())
            return JSC::jsNull();
        return toJS<IDLDOMString>(lexicalGlobalObject, value);
    }

    static JSC::JSValue convert(JSC::JSGlobalObject& lexicalGlobalObject, const URL& value)
    {
        if (value.isNull())
            return JSC::jsNull();
        return toJS<IDLDOMString>(lexicalGlobalObject, value);
    }
};

template<> struct JSConverter<IDLNullable<IDLByteString>> {
    static constexpr bool needsState = true;
    static constexpr bool needsGlobalObject = false;

    static JSC::JSValue convert(JSC::JSGlobalObject& lexicalGlobalObject, const String& value)
    {
        if (value.isNull())
            return JSC::jsNull();
        return toJS<IDLByteString>(lexicalGlobalObject, value);
    }

    static JSC::JSValue convert(JSC::JSGlobalObject& lexicalGlobalObject, const UncachedString& value)
    {
        if (value.string.isNull())
            return JSC::jsNull();
        return toJS<IDLByteString>(lexicalGlobalObject, value);
    }

    static JSC::JSValue convert(JSC::JSGlobalObject& lexicalGlobalObject, const OwnedString& value)
    {
        if (value.string.isNull())
            return JSC::jsNull();
        return toJS<IDLByteString>(lexicalGlobalObject, value);
    }

    static JSC::JSValue convert(JSC::JSGlobalObject& lexicalGlobalObject, const URL& value)
    {
        if (value.isNull())
            return JSC::jsNull();
        return toJS<IDLByteString>(lexicalGlobalObject, value);
    }
};

template<> struct JSConverter<IDLNullable<IDLUSVString>> {
    static constexpr bool needsState = true;
    static constexpr bool needsGlobalObject = false;

    static JSC::JSValue convert(JSC::JSGlobalObject& lexicalGlobalObject, const String& value)
    {
        if (value.isNull())
            return JSC::jsNull();
        return toJS<IDLUSVString>(lexicalGlobalObject, value);
    }

    static JSC::JSValue convert(JSC::JSGlobalObject& lexicalGlobalObject, const UncachedString& value)
    {
        if (value.string.isNull())
            return JSC::jsNull();
        return toJS<IDLUSVString>(lexicalGlobalObject, value);
    }

    static JSC::JSValue convert(JSC::JSGlobalObject& lexicalGlobalObject, const OwnedString& value)
    {
        if (value.string.isNull())
            return JSC::jsNull();
        return toJS<IDLUSVString>(lexicalGlobalObject, value);
    }

    static JSC::JSValue convert(JSC::JSGlobalObject& lexicalGlobalObject, const URL& value)
    {
        if (value.isNull())
            return JSC::jsNull();
        return toJS<IDLUSVString>(lexicalGlobalObject, value);
    }
};

template<> struct JSConverter<IDLNullable<IDLAtomStringAdaptor<IDLDOMString>>> {
    static constexpr bool needsState = true;
    static constexpr bool needsGlobalObject = false;

    static JSC::JSValue convert(JSC::JSGlobalObject& lexicalGlobalObject, const String& value)
    {
        if (value.isNull())
            return JSC::jsNull();
        return toJS<IDLDOMString>(lexicalGlobalObject, value);
    }

    static JSC::JSValue convert(JSC::JSGlobalObject& lexicalGlobalObject, const AtomString& value)
    {
        if (value.isNull())
            return JSC::jsNull();
        return toJS<IDLDOMString>(lexicalGlobalObject, value.string());
    }
};

} // namespace WebCore
