/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 12, 2022.
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
#include <mach/mach_init.h>
#include <mach/mach_vm.h>
#include <stdlib.h>

int
main(void)
{
	kern_return_t kr;
	mach_vm_address_t addr = 50ULL * 1024ULL * 1024ULL * 1024ULL;

	kr = mach_vm_allocate(current_task(), &addr, 4096, VM_FLAGS_FIXED);

	if (kr == KERN_SUCCESS) {
		return 0;
	} else {
		return 1;
	}
}
