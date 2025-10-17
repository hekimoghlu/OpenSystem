/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 24, 2022.
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
#include "JSTypedArrayConstructors.h"

#include "JSCInlines.h"

namespace JSC {

#undef MAKE_S_INFO
#define MAKE_S_INFO(type) \
    template<> const ClassInfo JS##type##Constructor::s_info = { "Function"_s, &JS##type##Constructor::Base::s_info, nullptr, nullptr, CREATE_METHOD_TABLE(JS##type##Constructor) }

MAKE_S_INFO(Int8Array);
MAKE_S_INFO(Int16Array);
MAKE_S_INFO(Int32Array);
MAKE_S_INFO(Uint8Array);
MAKE_S_INFO(Uint8ClampedArray);
MAKE_S_INFO(Uint16Array);
MAKE_S_INFO(Uint32Array);
MAKE_S_INFO(Float16Array);
MAKE_S_INFO(Float32Array);
MAKE_S_INFO(Float64Array);
MAKE_S_INFO(BigInt64Array);
MAKE_S_INFO(BigUint64Array);
MAKE_S_INFO(DataView);

} // namespace JSC

