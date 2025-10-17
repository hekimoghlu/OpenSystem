/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 16, 2024.
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
#ifndef _KXLD_H
#define _KXLD_H

#include <sys/types.h>
#include <mach/boolean.h>       // boolean_t
#include <mach/kern_return.h>   // kern_return_t
#include <mach/machine.h>       // cpu_type_t and cpu_subtype_t
#include <mach/vm_types.h>

#include "kxld_types.h"

/*******************************************************************************
* API
*******************************************************************************/

/*******************************************************************************
* Creates a state object for the linker.  A context must be created for each
* link thread and destroyed at the end of the thread's life.  A context should
* be reused for all links occuring in that link thread.
*   context             Returns a pointer to the new context object
*   allocate callback   Callback to allocate memory for the linked kext
*   log_callback        Callback for all kxld logging output
*   flags               Flags to control the behavior of kxld
*   cputype             The target arch's CPU type (0 for host arch)
*   cpusubtype          The target arch's CPU subtype (0 for host subtype)
*   pagesize            The target page size (0 for host page size)
*******************************************************************************/
kern_return_t kxld_create_context(
	KXLDContext **context,
	KXLDAllocateCallback allocate_callback,
	KXLDLoggingCallback log_callback,
	KXLDFlags flags,
	cpu_type_t cputype,
	cpu_subtype_t cpusubtype,
	vm_size_t pagesize)
__attribute__((nonnull(1), visibility("default")));

/*******************************************************************************
* Destroys a link context and frees all associated memory.  Should be called at
* the end of a link thread's life.
*******************************************************************************/
void kxld_destroy_context(
	KXLDContext *context)
__attribute__((nonnull, visibility("default")));

/*******************************************************************************
 * Links a kext against its dependencies, using a callback to allocate the memory
 * at which it will be located.
 * NOTE: The object data itself must be mmapped with PROT_WRITE and MAP_PRIVATE
 *   context             The KXLDContext object for the current link thread.
 *   file                The kext object file read into memory.
 *                       Supported formats: Mach-O, Mach-O64, Fat.
 *   size                The size of the kext in memory.  Must be nonzero.
 *   name                The name, usually the bundle identifier, of the kext
 *   callback_data       Data that is to be passed to the callback functions.
 *   dependencies        An array of pointers to the kexts upon which this kext
 *                       is dependent.
 *   num_dependencies    Number of entries in the 'dependencies' array.
 *   linked_object       This will be set to the address of the linked kext
 *                       object. If the address provided by the
 *                       kxld_alloc_callback is considered writable, this
 *                       pointer will be set to that address.  Otherwise, the
 *                       linked object will be written to a temporary buffer
 *                       that should be freed by the caller.
 *   kmod_info_kern      Kernel address of the kmod_info_t structure.
 ******************************************************************************/
kern_return_t kxld_link_file(
	KXLDContext *context,
	u_char *file,
	u_long size,
	const char *name,
	void *callback_data,
	KXLDDependency *dependencies,
	u_int num_dependencies,
	u_char **linked_object,
	kxld_addr_t *kmod_info_kern)
__attribute__((nonnull(1, 2, 4, 6, 8, 9), visibility("default")));


kern_return_t kxld_link_split_file(
	KXLDContext *context,
	splitKextLinkInfo *link_info,
	const char *name,
	void *callback_data,
	KXLDDependency *dependencies,
	u_int num_dependencies,
	kxld_addr_t *kmod_info_kern)
__attribute__((nonnull(1, 2, 3, 5, 7), visibility("default")));


/*******************************************************************************
*******************************************************************************/
boolean_t kxld_validate_copyright_string(const char *str)
__attribute__((pure, nonnull, visibility("default")));

#endif // _KXLD_H_
