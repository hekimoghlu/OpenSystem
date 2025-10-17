/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 14, 2024.
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
#ifndef _UAPI_LINUX_STDDEF_H
#define _UAPI_LINUX_STDDEF_H
#include <linux/compiler_types.h>
#ifndef __always_inline
#define __always_inline inline
#endif
#define __struct_group(TAG,NAME,ATTRS,MEMBERS...) union { struct { MEMBERS } ATTRS; struct TAG { MEMBERS } ATTRS NAME; } ATTRS
#ifdef __cplusplus
#define __DECLARE_FLEX_ARRAY(T,member) T member[0]
#else
#define __DECLARE_FLEX_ARRAY(TYPE,NAME) struct { struct { } __empty_ ##NAME; TYPE NAME[]; }
#endif
#ifndef __counted_by
#define __counted_by(m)
#endif
#ifndef __counted_by_le
#define __counted_by_le(m)
#endif
#ifndef __counted_by_be
#define __counted_by_be(m)
#endif
#endif
