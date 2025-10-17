/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 11, 2024.
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
#include <err.h>
#include <crt_externs.h>
#include <string.h>
#include <mach/mach.h>
#include <mach-o/ldsyms.h>
#include <mach-o/dyld_images.h>
#include <mach-o/arch.h>
#include <stdlib.h>
#include <sys/sysctl.h>

__attribute__((constructor))
void
init(int argc, const char *argv[], const char *envp[], const char *appl[], void *vars __attribute__((unused)))
{
	int i;

	printf("argv = %p\n", argv);
	for (i = 0; argv[i]; i++) {
		printf("argv[%2d] = %p %.100s%s\n", i, argv[i], argv[i], strlen(argv[i]) > 100 ? "..." : "");
	}
	printf("envp = %p\n", envp);
	for (i = 0; envp[i]; i++) {
		printf("envp[%2d] = %p %.100s%s\n", i, envp[i], envp[i], strlen(envp[i]) > 100 ? "..." : "");
	}
	printf("appl = %p\n", appl);
	for (i = 0; appl[i]; i++) {
		printf("appl[%2d] = %p %.100s%s\n", i, appl[i], appl[i], strlen(appl[i]) > 100 ? "..." : "");
	}
}

void
printexecinfo(void)
{
	int ret;
	uint64_t stackaddr;
	size_t len = sizeof(stackaddr);
	const NXArchInfo *arch = NXGetArchInfoFromCpuType(_mh_execute_header.cputype, _mh_execute_header.cpusubtype & ~CPU_SUBTYPE_MASK);

	printf("executable load address = 0x%016llx\n", (uint64_t)(uintptr_t)&_mh_execute_header);
	printf("executable cputype 0x%08x cpusubtype 0x%08x (%s:%s)\n",
	    _mh_execute_header.cputype,
	    _mh_execute_header.cpusubtype,
	    arch ? arch->name : "unknown",
	    arch ? arch->description : "unknown");

	ret = sysctlbyname("kern.usrstack64", &stackaddr, &len, NULL, 0);
	if (ret == -1) {
		err(1, "sysctlbyname");
	}

	printf("          stack address = 0x%016llx\n", stackaddr);
}

void
printdyldinfo(void)
{
	task_dyld_info_data_t info;
	mach_msg_type_number_t size = TASK_DYLD_INFO_COUNT;
	kern_return_t kret;
	struct dyld_all_image_infos *all_image_infos;

	kret = task_info(mach_task_self(), TASK_DYLD_INFO,
	    (void *)&info, &size);
	if (kret != KERN_SUCCESS) {
		errx(1, "task_info: %s", mach_error_string(kret));
	}

	all_image_infos = (struct dyld_all_image_infos *)(uintptr_t)info.all_image_info_addr;

	printf("      dyld load address = 0x%016llx\n", (uint64_t)(uintptr_t)all_image_infos->dyldImageLoadAddress);
	printf("     shared cache slide = 0x%016llx\n", (uint64_t)(uintptr_t)all_image_infos->sharedCacheSlide);
}

int
main(int argc, char *argv[])
{
	printexecinfo();
	printdyldinfo();

	return 0;
}
