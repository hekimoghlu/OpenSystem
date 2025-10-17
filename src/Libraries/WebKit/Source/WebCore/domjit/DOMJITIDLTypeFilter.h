/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 7, 2022.
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
#include <JavaScriptCore/SpeculatedType.h>

namespace WebCore { namespace DOMJIT {

template<typename IDLType>
struct IDLArgumentTypeFilter;

template<> struct IDLArgumentTypeFilter<IDLBoolean> { static const constexpr JSC::SpeculatedType value = JSC::SpecBoolean; };
template<> struct IDLArgumentTypeFilter<IDLByte> { static const constexpr JSC::SpeculatedType value = JSC::SpecInt32Only; };
template<> struct IDLArgumentTypeFilter<IDLOctet> { static const constexpr JSC::SpeculatedType value = JSC::SpecInt32Only; };
template<> struct IDLArgumentTypeFilter<IDLShort> { static const constexpr JSC::SpeculatedType value = JSC::SpecInt32Only; };
template<> struct IDLArgumentTypeFilter<IDLUnsignedShort> { static const constexpr JSC::SpeculatedType value = JSC::SpecInt32Only; };
template<> struct IDLArgumentTypeFilter<IDLLong> { static const constexpr JSC::SpeculatedType value = JSC::SpecInt32Only; };
template<> struct IDLArgumentTypeFilter<IDLDOMString> { static const constexpr JSC::SpeculatedType value = JSC::SpecString; };
template<> struct IDLArgumentTypeFilter<IDLAtomStringAdaptor<IDLDOMString>> { static const constexpr JSC::SpeculatedType value = JSC::SpecString; };
template<> struct IDLArgumentTypeFilter<IDLRequiresExistingAtomStringAdaptor<IDLDOMString>> { static const constexpr JSC::SpeculatedType value = JSC::SpecString; };

template<typename IDLType>
struct IDLResultTypeFilter {
    static const constexpr JSC::SpeculatedType value = JSC::SpecHeapTop;
};

template<> struct IDLResultTypeFilter<IDLAny> { static const constexpr JSC::SpeculatedType value = JSC::SpecHeapTop; };
template<> struct IDLResultTypeFilter<IDLBoolean> { static const constexpr JSC::SpeculatedType value = JSC::SpecBoolean; };
template<> struct IDLResultTypeFilter<IDLByte> { static const constexpr JSC::SpeculatedType value = JSC::SpecInt32Only; };
template<> struct IDLResultTypeFilter<IDLOctet> { static const constexpr JSC::SpeculatedType value = JSC::SpecInt32Only; };
template<> struct IDLResultTypeFilter<IDLShort> { static const constexpr JSC::SpeculatedType value = JSC::SpecInt32Only; };
template<> struct IDLResultTypeFilter<IDLUnsignedShort> { static const constexpr JSC::SpeculatedType value = JSC::SpecInt32Only; };
template<> struct IDLResultTypeFilter<IDLLong> { static const constexpr JSC::SpeculatedType value = JSC::SpecInt32Only; };
template<> struct IDLResultTypeFilter<IDLUnsignedLong> { static const constexpr JSC::SpeculatedType value = JSC::SpecBytecodeNumber; };
template<> struct IDLResultTypeFilter<IDLLongLong> { static const constexpr JSC::SpeculatedType value = JSC::SpecBytecodeNumber; };
template<> struct IDLResultTypeFilter<IDLUnsignedLongLong> { static const constexpr JSC::SpeculatedType value = JSC::SpecBytecodeNumber; };
template<> struct IDLResultTypeFilter<IDLFloat> { static const constexpr JSC::SpeculatedType value = JSC::SpecBytecodeNumber; };
template<> struct IDLResultTypeFilter<IDLUnrestrictedFloat> { static const constexpr JSC::SpeculatedType value = JSC::SpecBytecodeNumber; };
template<> struct IDLResultTypeFilter<IDLDouble> { static const constexpr JSC::SpeculatedType value = JSC::SpecBytecodeNumber; };
template<> struct IDLResultTypeFilter<IDLUnrestrictedDouble> { static const constexpr JSC::SpeculatedType value = JSC::SpecBytecodeNumber; };
template<> struct IDLResultTypeFilter<IDLDOMString> { static const constexpr JSC::SpeculatedType value = JSC::SpecString; };
template<> struct IDLResultTypeFilter<IDLByteString> { static const constexpr JSC::SpeculatedType value = JSC::SpecString; };
template<> struct IDLResultTypeFilter<IDLUSVString> { static const constexpr JSC::SpeculatedType value = JSC::SpecString; };
template<> struct IDLResultTypeFilter<IDLAtomStringAdaptor<IDLDOMString>> { static const constexpr JSC::SpeculatedType value = JSC::SpecString; };
template<> struct IDLResultTypeFilter<IDLRequiresExistingAtomStringAdaptor<IDLDOMString>> { static const constexpr JSC::SpeculatedType value = JSC::SpecString; };

template<typename T>
struct IDLResultTypeFilter<IDLNullable<T>> {
    static const constexpr JSC::SpeculatedType value = JSC::SpecOther | IDLResultTypeFilter<T>::value;
};

} }
