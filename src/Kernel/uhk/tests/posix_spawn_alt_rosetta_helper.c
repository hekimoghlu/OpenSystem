/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 1, 2023.
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
#include <stdio.h>
#include <mach-o/dyld.h>
#include <mach-o/dyld_images.h>
#include <libproc.h>

/*
 * Returns 1 if the standard Rosetta runtime is loaded, 2 if the alternative
 * runtime is loaded, and 0 if no runtime was detected.
 */
int
main(int argc, const char * argv[])
{
	unsigned depth = 1;
	vm_size_t size = 0;
	vm_address_t address = 0;
	vm_address_t end_address;
	kern_return_t err = KERN_SUCCESS;
	mach_msg_type_number_t count = VM_REGION_SUBMAP_INFO_COUNT_64;
	struct vm_region_submap_info_64 info;
	char path_buffer[MAXPATHLEN + 1] = {0};

	while (1) {
		err = vm_region_recurse_64(mach_task_self(), &address, &size, &depth, (vm_region_info_t)&info, &count);
		if (err != KERN_SUCCESS) {
			break;
		}

		end_address = address + size;
		err = proc_regionfilename(getpid(), address, path_buffer, MAXPATHLEN);
		if (err == KERN_SUCCESS) {
			if (strcmp(path_buffer, "/usr/libexec/rosetta/runtime") == 0) {
				printf("0x%016lx-0x%016lx %s\n", address, end_address, path_buffer);
				return 1;
			} else if (strcmp(path_buffer, "/usr/local/libexec/rosetta/runtime_internal") == 0) {
				printf("0x%016lx-0x%016lx %s\n", address, end_address, path_buffer);
				return 2;
			}
		}

		address = end_address;
	}
	return 0;
}
