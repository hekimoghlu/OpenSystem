/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 1, 2022.
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
#ifndef _ARM_PAL_ROUTINES_H
#define _ARM_PAL_ROUTINES_H

#include <stdint.h>
#include <string.h>

#if defined(__cplusplus)
extern "C" {
#endif

#ifdef XNU_KERNEL_PRIVATE

/* No-op */
#define pal_dbg_set_task_name( x ) do { } while(0)

#define pal_ast_check(thread)
#define pal_thread_terminate_self(t)

/* serial / debug output routines */
extern int  pal_serial_init(void);
extern void pal_serial_putc(char a);
extern void pal_serial_putc_nocr(char a);
extern int  pal_serial_getc(void);

#define panic_display_pal_info() do { } while(0)
#define pal_kernel_announce() do { } while(0)

#endif  /* XNU_KERNEL_PRIVATE */

/* Allows us to set a property on the IOResources object. Unused on ARM. */
static inline void
pal_get_resource_property(const char **property_name,
    int *property_value)
{
	*property_name = NULL;
	(void) property_value;
}

#if defined(__cplusplus)
}
#endif

#endif /* _ARM_PAL_ROUTINES_H */
