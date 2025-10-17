/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 19, 2022.
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
#ifndef _OS_PTRTOOLS_H
#define _OS_PTRTOOLS_H

/* dereference unaligned pointer 'p' */
#define os_unaligned_deref(p) ((__os_unaligned_type(p))(p))->val

/* ensure the compiler emits at most one access to 'val' */
#define os_access_once(val) (*((volatile __typeof__((val)) *)&(val)))

#define __os_unaligned_type(p) struct { __typeof__(*(p)) val; } __attribute__((packed)) *

#endif /* _OS_PTRTOOLS_H */
