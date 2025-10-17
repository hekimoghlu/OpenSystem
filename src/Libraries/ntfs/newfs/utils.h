/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 6, 2023.
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
#ifndef _NTFS_UTILS_H_
#define _NTFS_UTILS_H_

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "types.h"
#include "layout.h"
#include "volume.h"

#ifdef HAVE_ERRNO_H
#include <errno.h>
#endif
#ifdef HAVE_STDARG_H
#include <stdarg.h>
#endif

int utils_set_locale(void);

/**
 * linux-ntfs's ntfs_mbstoucs has different semantics, so we emulate it with
 * ntfs-3g's.
 */
int ntfs_mbstoucs_libntfscompat(const char *ins,
		ntfschar **outs, int outs_len);

#endif /* _NTFS_UTILS_H_ */
