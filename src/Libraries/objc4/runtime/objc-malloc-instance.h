/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 19, 2025.
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
#ifndef _OBJC_MALLOC_INSTANCE_H
#define _OBJC_MALLOC_INSTANCE_H

#include <cstdlib>
#if _MALLOC_TYPE_ENABLED
# include <malloc_type_private.h>
#endif

namespace objc {

static inline id
malloc_instance(size_t size, Class cls __unused)
{
#if _MALLOC_TYPE_ENABLED
    malloc_type_descriptor_t desc = {};
    desc.summary.type_kind = MALLOC_TYPE_KIND_OBJC;
    return (id)malloc_type_calloc(1, size, desc.type_id);
#else
    return (id)calloc(1, size);
#endif
}

} // namespace objc

#endif // _OBJC_MALLOC_INSTANCE_H
