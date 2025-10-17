/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 21, 2022.
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
#ifndef __AUTOFS_TYPES_H__
#define __AUTOFS_TYPES_H__

#include <sys/syslimits.h>
#include <sys/mount.h>          /* for fsid_t */
#include "autofs_defs.h"

/*
 * Type definitions, C-style.
 */
typedef char autofs_pathname[PATH_MAX + 1];
typedef char autofs_component[NAME_MAX];        /* not null-terminated! */
typedef char autofs_fstype[NAME_MAX + 1];
typedef char autofs_opts[AUTOFS_MAXOPTSLEN];
typedef uint8_t *byte_buffer;

#endif /* __AUTOFS_TYPES_H__ */
