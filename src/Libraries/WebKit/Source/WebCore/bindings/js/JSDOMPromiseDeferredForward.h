/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 16, 2024.
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

namespace WebCore {

class DeferredPromise;

template <typename IDLType> class DOMPromiseDeferred;
template <typename IDLType> class DOMPromiseProxy;

struct IDLUnsupportedType;
struct IDLNull;
struct IDLAny;
struct IDLUndefined;
struct IDLBoolean;
struct IDLByte;
struct IDLOctet;
struct IDLShort;
struct IDLUnsignedShort;
struct IDLLong;
struct IDLUnsignedLong;
struct IDLLongLong;
struct IDLUnsignedLongLong;
struct IDLFloat;
struct IDLUnrestrictedFloat;
struct IDLDouble;
struct IDLUnrestrictedDouble;
struct IDLDOMString;
struct IDLByteString;
struct IDLUSVString;
struct IDLObject;

template<typename> struct IDLInterface;
template<typename> struct IDLCallbackInterface;
template<typename> struct IDLCallbackFunction;
template<typename> struct IDLDictionary;
template<typename> struct IDLEnumeration;
template<typename> struct IDLNullable;
template<typename> struct IDLSequence;
template<typename> struct IDLFrozenArray;
template<typename, typename> struct IDLRecord;
template<typename> struct IDLPromise;
template<typename> struct IDLPromiseIgnoringSuspension;

struct IDLError;
struct IDLDOMException;

template<typename...> struct IDLUnion;
template<typename> struct IDLBufferSourceBase;

struct IDLArrayBuffer;
struct IDLArrayBufferView;
struct IDLDataView;
struct IDLDate;
struct IDLJSON;
struct IDLScheduledAction;
struct IDLIDBKey;
struct IDLIDBKeyData;
struct IDLIDBValue;

#if ENABLE(WEBGL)
struct IDLWebGLAny;
struct IDLWebGLExtensionAny;
#endif

}
