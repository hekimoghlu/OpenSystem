/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 4, 2022.
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
#ifndef _NTFS_DEBUG_H
#define _NTFS_DEBUG_H

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "logging.h"

struct _runlist_element;

#ifdef DEBUG
extern void ntfs_debug_runlist_dump(const struct _runlist_element *rl);
#else
static __inline__ void ntfs_debug_runlist_dump(const struct _runlist_element *rl __attribute__((unused))) {}
#endif

#define NTFS_BUG(msg)							\
{									\
	int ___i = 1;							\
	ntfs_log_critical("Bug in %s(): %s\n", __FUNCTION__, msg);	\
	ntfs_log_debug("Forcing segmentation fault!");			\
	___i = ((int*)NULL)[___i];					\
}

#endif /* defined _NTFS_DEBUG_H */
