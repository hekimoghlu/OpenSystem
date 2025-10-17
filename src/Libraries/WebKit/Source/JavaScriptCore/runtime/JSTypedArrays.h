/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 24, 2023.
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

#include "JSDataView.h"
#include "JSGenericTypedArrayView.h"
#include "TypedArrayAdaptors.h"

namespace JSC {

using JSInt8Array = JSGenericTypedArrayView<Int8Adaptor>;
using JSInt16Array = JSGenericTypedArrayView<Int16Adaptor>;
using JSInt32Array = JSGenericTypedArrayView<Int32Adaptor>;
using JSUint8Array = JSGenericTypedArrayView<Uint8Adaptor>;
using JSUint8ClampedArray = JSGenericTypedArrayView<Uint8ClampedAdaptor>;
using JSUint16Array = JSGenericTypedArrayView<Uint16Adaptor>;
using JSUint32Array = JSGenericTypedArrayView<Uint32Adaptor>;
using JSFloat16Array = JSGenericTypedArrayView<Float16Adaptor>;
using JSFloat32Array = JSGenericTypedArrayView<Float32Adaptor>;
using JSFloat64Array = JSGenericTypedArrayView<Float64Adaptor>;
using JSBigInt64Array = JSGenericTypedArrayView<BigInt64Adaptor>;
using JSBigUint64Array = JSGenericTypedArrayView<BigUint64Adaptor>;
using JSResizableOrGrowableSharedInt8Array = JSGenericResizableOrGrowableSharedTypedArrayView<Int8Adaptor>;
using JSResizableOrGrowableSharedInt16Array = JSGenericResizableOrGrowableSharedTypedArrayView<Int16Adaptor>;
using JSResizableOrGrowableSharedInt32Array = JSGenericResizableOrGrowableSharedTypedArrayView<Int32Adaptor>;
using JSResizableOrGrowableSharedUint8Array = JSGenericResizableOrGrowableSharedTypedArrayView<Uint8Adaptor>;
using JSResizableOrGrowableSharedUint8ClampedArray = JSGenericResizableOrGrowableSharedTypedArrayView<Uint8ClampedAdaptor>;
using JSResizableOrGrowableSharedUint16Array = JSGenericResizableOrGrowableSharedTypedArrayView<Uint16Adaptor>;
using JSResizableOrGrowableSharedUint32Array = JSGenericResizableOrGrowableSharedTypedArrayView<Uint32Adaptor>;
using JSResizableOrGrowableSharedFloat16Array = JSGenericResizableOrGrowableSharedTypedArrayView<Float16Adaptor>;
using JSResizableOrGrowableSharedFloat32Array = JSGenericResizableOrGrowableSharedTypedArrayView<Float32Adaptor>;
using JSResizableOrGrowableSharedFloat64Array = JSGenericResizableOrGrowableSharedTypedArrayView<Float64Adaptor>;
using JSResizableOrGrowableSharedBigInt64Array = JSGenericResizableOrGrowableSharedTypedArrayView<BigInt64Adaptor>;
using JSResizableOrGrowableSharedBigUint64Array = JSGenericResizableOrGrowableSharedTypedArrayView<BigUint64Adaptor>;

inline bool isResizableOrGrowableSharedTypedArrayIncludingDataView(const ClassInfo* classInfo)
{
    return classInfo->isResizableOrGrowableSharedTypedArray;
}

} // namespace JSC
