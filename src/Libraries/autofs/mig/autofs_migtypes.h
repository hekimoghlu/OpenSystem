/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 4, 2023.
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
#ifndef __AUTOFS_MIGTYPES_H__
#define __AUTOFS_MIGTYPES_H__

/*
 * Type definitions, MIG-style.
 */
type autofs_pathname = c_string[*:PATH_MAX];
type autofs_component = array[*:NAME_MAX] of char;      /* not null-terminated! */
type autofs_fstype = c_string[*:NAME_MAX];
type autofs_opts = c_string[*:AUTOFS_MAXOPTSLEN];
type byte_buffer = array[] of uint8_t;

/*
 * Fortunately, an fsid_t looks like a structure with 2 32-bit int
 * members (it's really a structure with one member that is an array
 * of 2 32-bit ints), so that's how we describe it to MIG.
 */
type fsid_t = struct[2] of int32_t;

#endif /* __AUTOFS_MIGTYPES_H__ */
