/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 5, 2024.
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

#include "JSGenericTypedArrayViewPrototype.h"
#include "JSTypedArrays.h"

namespace JSC {

typedef JSGenericTypedArrayViewPrototype<JSInt8Array> JSInt8ArrayPrototype;
typedef JSGenericTypedArrayViewPrototype<JSInt16Array> JSInt16ArrayPrototype;
typedef JSGenericTypedArrayViewPrototype<JSInt32Array> JSInt32ArrayPrototype;
typedef JSGenericTypedArrayViewPrototype<JSUint8Array> JSUint8ArrayPrototype;
typedef JSGenericTypedArrayViewPrototype<JSUint8ClampedArray> JSUint8ClampedArrayPrototype;
typedef JSGenericTypedArrayViewPrototype<JSUint16Array> JSUint16ArrayPrototype;
typedef JSGenericTypedArrayViewPrototype<JSUint32Array> JSUint32ArrayPrototype;
typedef JSGenericTypedArrayViewPrototype<JSFloat16Array> JSFloat16ArrayPrototype;
typedef JSGenericTypedArrayViewPrototype<JSFloat32Array> JSFloat32ArrayPrototype;
typedef JSGenericTypedArrayViewPrototype<JSFloat64Array> JSFloat64ArrayPrototype;
typedef JSGenericTypedArrayViewPrototype<JSBigInt64Array> JSBigInt64ArrayPrototype;
typedef JSGenericTypedArrayViewPrototype<JSBigUint64Array> JSBigUint64ArrayPrototype;

} // namespace JSC
