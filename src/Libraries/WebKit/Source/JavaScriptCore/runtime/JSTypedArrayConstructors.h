/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 2, 2022.
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
#include "JSGenericTypedArrayViewConstructor.h"
#include "JSTypedArrays.h"

namespace JSC {

using JSInt8ArrayConstructor = JSGenericTypedArrayViewConstructor<JSInt8Array>;
using JSInt16ArrayConstructor = JSGenericTypedArrayViewConstructor<JSInt16Array>;
using JSInt32ArrayConstructor = JSGenericTypedArrayViewConstructor<JSInt32Array>;
using JSUint8ArrayConstructor = JSGenericTypedArrayViewConstructor<JSUint8Array>;
using JSUint8ClampedArrayConstructor = JSGenericTypedArrayViewConstructor<JSUint8ClampedArray>;
using JSUint16ArrayConstructor = JSGenericTypedArrayViewConstructor<JSUint16Array>;
using JSUint32ArrayConstructor = JSGenericTypedArrayViewConstructor<JSUint32Array>;
using JSFloat16ArrayConstructor = JSGenericTypedArrayViewConstructor<JSFloat16Array>;
using JSFloat32ArrayConstructor = JSGenericTypedArrayViewConstructor<JSFloat32Array>;
using JSFloat64ArrayConstructor = JSGenericTypedArrayViewConstructor<JSFloat64Array>;
using JSBigInt64ArrayConstructor = JSGenericTypedArrayViewConstructor<JSBigInt64Array>;
using JSBigUint64ArrayConstructor = JSGenericTypedArrayViewConstructor<JSBigUint64Array>;
using JSDataViewConstructor = JSGenericTypedArrayViewConstructor<JSDataView>;
STATIC_ASSERT_ISO_SUBSPACE_SHARABLE(JSInt8ArrayConstructor, InternalFunction);
STATIC_ASSERT_ISO_SUBSPACE_SHARABLE(JSInt16ArrayConstructor, InternalFunction);
STATIC_ASSERT_ISO_SUBSPACE_SHARABLE(JSInt32ArrayConstructor, InternalFunction);
STATIC_ASSERT_ISO_SUBSPACE_SHARABLE(JSUint8ArrayConstructor, InternalFunction);
STATIC_ASSERT_ISO_SUBSPACE_SHARABLE(JSUint8ClampedArrayConstructor, InternalFunction);
STATIC_ASSERT_ISO_SUBSPACE_SHARABLE(JSUint16ArrayConstructor, InternalFunction);
STATIC_ASSERT_ISO_SUBSPACE_SHARABLE(JSUint32ArrayConstructor, InternalFunction);
STATIC_ASSERT_ISO_SUBSPACE_SHARABLE(JSFloat16ArrayConstructor, InternalFunction);
STATIC_ASSERT_ISO_SUBSPACE_SHARABLE(JSFloat32ArrayConstructor, InternalFunction);
STATIC_ASSERT_ISO_SUBSPACE_SHARABLE(JSFloat64ArrayConstructor, InternalFunction);
STATIC_ASSERT_ISO_SUBSPACE_SHARABLE(JSBigInt64ArrayConstructor, InternalFunction);
STATIC_ASSERT_ISO_SUBSPACE_SHARABLE(JSBigUint64ArrayConstructor, InternalFunction);
STATIC_ASSERT_ISO_SUBSPACE_SHARABLE(JSDataViewConstructor, InternalFunction);

} // namespace JSC
