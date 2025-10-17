/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 9, 2022.
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
#ifndef _I386_ACPI_H_
#define _I386_ACPI_H_

/*
 * ACPI (Advanced Configuration and Power Interface) support.
 */

/*
 * Wake up code linear address. Wake and MP startup copy
 * code to this physical address and then jump to the
 * address started at PROT_MODE_START. Some small amount
 * below PROT_MODE_START is used as scratch space
 */
#define PROT_MODE_START 0x800
#define REAL_MODE_BOOTSTRAP_OFFSET 0x2000

#ifndef ASSEMBLER
typedef void (*acpi_sleep_callback)(void * refcon);
extern vm_offset_t acpi_install_wake_handler(void);
extern void        acpi_sleep_kernel(acpi_sleep_callback func, void * refcon);
extern void        acpi_idle_kernel(acpi_sleep_callback func, void * refcon);
void install_real_mode_bootstrap(void *prot_entry);
extern uint32_t    acpi_count_enabled_logical_processors(void);
#endif  /* ASSEMBLER */

#endif /* !_I386_ACPI_H_ */
