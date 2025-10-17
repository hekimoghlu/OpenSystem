/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 26, 2022.
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
/*
 * Mach Operating System
 * Copyright (c) 1990 Carnegie-Mellon University
 * All rights reserved.  The CMU software License Agreement specifies
 * the terms and conditions for use and redistribution.
 */
/*
 *	File:	hostinfo.c
 *	Author:	Avadis Tevanian, Jr.
 *
 *	Copyright (C) 1987, Avadis Tevanian, Jr.
 *
 *	Display information about the host this program is
 *	execting on.
 */

#include <mach/mach.h>
#include <mach/mach_error.h>
#include <sys/sysctl.h>
#include <sys/errno.h>
#include <stdio.h>
#include <stdlib.h>

struct host_basic_info	hi;
kernel_version_t	version;
int			slots[1024];

int
main(int argc, char *argv[])
{
	kern_return_t		ret;
	unsigned int		size, count;
	char			*cpu_name, *cpu_subname;
	int			mib[2];
	size_t			len;
	uint64_t		memsize;
	processor_set_name_port_t		default_pset;
	host_name_port_t			host;
	struct processor_set_basic_info	basic_info;
	struct processor_set_load_info	load_info;

	host = mach_host_self();
	ret = host_kernel_version(host, version);
	if (ret != KERN_SUCCESS) {
		mach_error(argv[0], ret);
                exit(EXIT_FAILURE);
	}
	printf("Mach kernel version:\n\t %s\n", version);
	size = sizeof(hi)/sizeof(int);
	ret = host_info(host, HOST_BASIC_INFO, (host_info_t)&hi, &size);
	if (ret != KERN_SUCCESS) {
		mach_error(argv[0], ret);
                exit(EXIT_FAILURE);
	}

	ret = processor_set_default(host, &default_pset);
	if (ret != KERN_SUCCESS) {
		mach_error(argv[0], ret);
                exit(EXIT_FAILURE);
	}

	count = PROCESSOR_SET_BASIC_INFO_COUNT;
	ret = processor_set_info(default_pset, PROCESSOR_SET_BASIC_INFO,
		&host, (processor_set_info_t)&basic_info, &count);
	if (ret != KERN_SUCCESS) {
		mach_error(argv[0], ret);
                exit(EXIT_FAILURE);
	}

	count = PROCESSOR_SET_LOAD_INFO_COUNT;
	ret = processor_set_statistics(default_pset, PROCESSOR_SET_LOAD_INFO,
		(processor_set_info_t)&load_info, &count);
	if (ret != KERN_SUCCESS) {
		mach_error(argv[0], ret);
                exit(EXIT_FAILURE);
	}

	unsigned int cpu_count = 0;
	unsigned int data_count = 0;
	struct processor_basic_info *processor_basic_infop = NULL;
	ret = host_processor_info(host,
	                          PROCESSOR_BASIC_INFO,
	                          &cpu_count,
	                          (processor_info_array_t *)&processor_basic_infop,
	                          &data_count);
	if (ret != KERN_SUCCESS) {
		mach_error(argv[0], ret);
		exit(EXIT_FAILURE);
	}

	mib[0] = CTL_HW;
	mib[1] = HW_MEMSIZE;
	len = sizeof(memsize);
	memsize = 0L;
	if(sysctl(mib, 2, &memsize, &len, NULL, 0 ) == -1)
	{
	    perror("sysctl");
	    exit(EXIT_FAILURE);
	}

	if (hi.max_cpus > 1)
		printf("Kernel configured for up to %d processors.\n",
			hi.max_cpus);
	else
		printf("Kernel configured for a single processor only.\n");
	printf("%d processor%s physically available.\n", hi.physical_cpu,
		(hi.physical_cpu > 1) ? "s are" : " is");

	printf("%d processor%s logically available.\n", hi.logical_cpu,
		(hi.logical_cpu > 1) ? "s are" : " is");

	printf("Processor type:");
	slot_name(hi.cpu_type, hi.cpu_subtype, &cpu_name, &cpu_subname);
	printf(" %s (%s)\n", cpu_name, cpu_subname);

	printf("Processor%s active:", (hi.avail_cpus > 1) ? "s" : "");
	for (int i = 0; i < cpu_count; i++) {
		if (processor_basic_infop[i].running) {
			printf(" %d", i);
		}
	}
	printf("\n");

	if (((float)memsize / (1024.0 * 1024.0)) >= 1024.0)
	    printf("Primary memory available: %.2f gigabytes\n",
	      (float)memsize/(1024.0*1024.0*1024.0));
	else
	    printf("Primary memory available: %.2f megabytes\n",
	      (float)memsize/(1024.0*1024.0));

	printf("Default processor set: %d tasks, %d threads, %d processors\n",
		load_info.task_count, load_info.thread_count, basic_info.processor_count);
	printf("Load average: %d.%02d, Mach factor: %d.%02d\n",
		load_info.load_average/LOAD_SCALE,
		(load_info.load_average%LOAD_SCALE)/10,
		load_info.mach_factor/LOAD_SCALE,
		(load_info.mach_factor%LOAD_SCALE)/10);

	exit(0);
}
