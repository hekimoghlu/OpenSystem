/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 24, 2022.
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
/**
 * i386/x86_64 specific definitions for hibernation platform abstraction layer.
 */

#ifndef _I386_PAL_HIBERNATE_H
#define _I386_PAL_HIBERNATE_H

__BEGIN_DECLS

#define HIB_MAP_SIZE    (2*I386_LPGBYTES)
#define DEST_COPY_AREA  (4*GB - HIB_MAP_SIZE) /*4GB - 2*2m */
#define SRC_COPY_AREA   (DEST_COPY_AREA - HIB_MAP_SIZE)
#define COPY_PAGE_AREA  (SRC_COPY_AREA  - HIB_MAP_SIZE)
#define BITMAP_AREA     (COPY_PAGE_AREA - HIB_MAP_SIZE)
#define IMAGE_AREA      (BITMAP_AREA    - HIB_MAP_SIZE)
#define IMAGE2_AREA     (IMAGE_AREA     - HIB_MAP_SIZE)
#define SCRATCH_AREA    (IMAGE2_AREA    - HIB_MAP_SIZE)
#define WKDM_AREA       (SCRATCH_AREA   - HIB_MAP_SIZE)

#define HIB_BASE segHIBB
#define HIB_ENTRYPOINT acpi_wake_prot_entry

/*!
 * @typedef     pal_hib_map_type_t
 * @discussion  Parameter to pal_hib_map used to signify which memory region to map.
 */
typedef uintptr_t pal_hib_map_type_t;

/*!
 * @struct      pal_hib_ctx
 * @discussion  x86_64-specific PAL context; see pal_hib_ctx_t for details.
 */
struct pal_hib_ctx {
	char reserved;
};

__END_DECLS

#endif /* _I386_PAL_HIBERNATE_H */
