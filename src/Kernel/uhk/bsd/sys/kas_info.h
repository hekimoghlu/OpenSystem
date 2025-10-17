/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 15, 2024.
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
#ifndef _SYS_KAS_INFO_H_
#define _SYS_KAS_INFO_H_

#include <sys/cdefs.h>
#include <Availability.h>

/*
 * kas_info() ("Kernel Address Space Info") is a private interface that allows
 * appropriately privileged system components to introspect the overall
 * kernel address space layout.
 */

__BEGIN_DECLS

/* The slide of the main kernel compared to its static link address */
#define KAS_INFO_KERNEL_TEXT_SLIDE_SELECTOR     (0) /* returns uint64_t */
#define KAS_INFO_KERNEL_SEGMENT_VMADDR_SELECTOR (1)

/* Return the SPTM/TXM slide if on a system configured to run those images. */
#define KAS_INFO_SPTM_TEXT_SLIDE_SELECTOR       (2) /* returns uint64_t */
#define KAS_INFO_TXM_TEXT_SLIDE_SELECTOR        (3) /* returns uint64_t */
#define KAS_INFO_MAX_SELECTOR                   (4)

#ifndef KERNEL

int kas_info(int selector, void *value, size_t *size) __OSX_AVAILABLE_STARTING(__MAC_10_8, __IPHONE_6_0);

#endif /* KERNEL */

__END_DECLS

#endif  /* !_SYS_KAS_INFO_H_ */
