/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 28, 2022.
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
#ifndef _MACHINE_DISKLABEL_H_
#define _MACHINE_DISKLABEL_H_

#include <sys/appleapiopts.h>

#ifdef __APPLE_API_OBSOLETE
#define LABELSECTOR     (1024 / DEV_BSIZE)      /* sector containing label */
#define LABELOFFSET     0                       /* offset of label in sector */
#define MAXPARTITIONS   8                       /* number of partitions */
#define RAW_PART        2                       /* raw partition: xx?c */

/* Just a dummy */
struct cpu_disklabel {
	int     cd_dummy;                       /* must have one element. */
};
#endif /* __APPLE_API_OBSOLETE */

#endif /* _MACHINE_DISKLABEL_H_ */
