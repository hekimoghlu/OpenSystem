/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 21, 2022.
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
#ifndef _SYS_KPI_PRIVATE_H
#define _SYS_KPI_PRIVATE_H

/*
 * Assorted odds and ends for exported private KPI (internal use only)
 */

#ifdef KERNEL
#include <sys/types.h>

__BEGIN_DECLS

#ifdef KERNEL_PRIVATE

/* kernel-exported qsort */
void kx_qsort(void* array, size_t nm, size_t member_size, int (*)(const void *, const void *));

#endif  /* KERNEL_PRIVATE */

__END_DECLS


#endif  /* KERNEL  */
#endif /* !_SYS_KPI_PRIVATE_H */
