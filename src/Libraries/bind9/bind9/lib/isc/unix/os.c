/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 13, 2023.
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
/* $Id: os.c,v 1.18 2007/06/19 23:47:18 tbox Exp $ */

#include <config.h>

#include <isc/os.h>


#ifdef HAVE_SYSCONF

#include <unistd.h>

#ifndef __hpux
static inline long
sysconf_ncpus(void) {
#if defined(_SC_NPROCESSORS_ONLN)
	return sysconf((_SC_NPROCESSORS_ONLN));
#elif defined(_SC_NPROC_ONLN)
	return sysconf((_SC_NPROC_ONLN));
#else
	return (0);
#endif
}
#endif
#endif /* HAVE_SYSCONF */


#ifdef __hpux

#include <sys/pstat.h>

static inline int
hpux_ncpus(void) {
	struct pst_dynamic psd;
	if (pstat_getdynamic(&psd, sizeof(psd), 1, 0) != -1)
		return (psd.psd_proc_cnt);
	else
		return (0);
}

#endif /* __hpux */

#if defined(HAVE_SYS_SYSCTL_H) && defined(HAVE_SYSCTLBYNAME)
#include <sys/types.h>  /* for FreeBSD */
#include <sys/param.h>  /* for NetBSD */
#include <sys/sysctl.h>

static int
sysctl_ncpus(void) {
	int ncpu, result;
	size_t len;

	len = sizeof(ncpu);
	result = sysctlbyname("hw.ncpu", &ncpu, &len , 0, 0);
	if (result != -1)
		return (ncpu);
	return (0);
}
#endif

unsigned int
isc_os_ncpus(void) {
	long ncpus = 0;

#ifdef __hpux
	ncpus = hpux_ncpus();
#elif defined(HAVE_SYSCONF)
	ncpus = sysconf_ncpus();
#endif
#if defined(HAVE_SYS_SYSCTL_H) && defined(HAVE_SYSCTLBYNAME)
	if (ncpus <= 0)
		ncpus = sysctl_ncpus();
#endif
	if (ncpus <= 0)
		ncpus = 1;

	return ((unsigned int)ncpus);
}
