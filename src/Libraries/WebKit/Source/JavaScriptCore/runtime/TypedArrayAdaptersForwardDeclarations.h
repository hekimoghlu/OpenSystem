/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 15, 2025.
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

namespace JSC {

struct Int8Adaptor;
struct Int16Adaptor;
struct Int32Adaptor;
struct Uint8Adaptor;
struct Uint8ClampedAdaptor;
struct Uint16Adaptor;
struct Uint32Adaptor;
struct Float16Adaptor;
struct Float32Adaptor;
struct Float64Adaptor;
struct BigInt64Adaptor;
struct BigUint64Adaptor;

template<typename Adaptor> class GenericTypedArrayView;
typedef GenericTypedArrayView<Int8Adaptor> Int8Array;
typedef GenericTypedArrayView<Int16Adaptor> Int16Array;
typedef GenericTypedArrayView<Int32Adaptor> Int32Array;
typedef GenericTypedArrayView<Uint8Adaptor> Uint8Array;
typedef GenericTypedArrayView<Uint8ClampedAdaptor> Uint8ClampedArray;
typedef GenericTypedArrayView<Uint16Adaptor> Uint16Array;
typedef GenericTypedArrayView<Uint32Adaptor> Uint32Array;
typedef GenericTypedArrayView<Float16Adaptor> Float16Array;
typedef GenericTypedArrayView<Float32Adaptor> Float32Array;
typedef GenericTypedArrayView<Float64Adaptor> Float64Array;
typedef GenericTypedArrayView<BigInt64Adaptor> BigInt64Array;
typedef GenericTypedArrayView<BigUint64Adaptor> BigUint64Array;

}

using JSC::Int8Array;
using JSC::Int16Array;
using JSC::Int32Array;
using JSC::Uint8Array;
using JSC::Uint8ClampedArray;
using JSC::Uint16Array;
using JSC::Uint32Array;
using JSC::Float16Array;
using JSC::Float32Array;
using JSC::Float64Array;
using JSC::BigInt64Array;
using JSC::BigUint64Array;
