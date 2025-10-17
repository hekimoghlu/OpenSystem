/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 24, 2022.
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
#include <string.h>

int
kvm_close(void* kd)
{
	return (0);
}

char**
kvm_getargv(void* kd, const void* p, int nchr)
{
	return (0);
}

char**
kvm_getenvv(void* kd, const void* p, int nchr)
{
	return (0);
}

char*
kvm_geterr(void* kd)
{
	return (0);
}

int
kvm_getloadavg(void* kd, double loadagv[], int nelem)
{
	return (-1);
}

char*
kvm_getfiles(void* kd, int op, int arg, int* cnt)
{
	if (cnt) *cnt = 0;
	return (0);
}

void*
kvm_getprocs(void* kd, int op, int arg, int* cnt)
{
	if (cnt) *cnt = 0;
	return (0);
}

int
kvm_nlist(void* kd, void* nl)
{
	return (-1);
}

void*
kvm_open(const char* execfile, const char* corefile, const char* swapfile, int flags, const char* errstr)
{
	fprintf(stderr, "%s%s/dev/mem: No such file or directory", errstr ? errstr : "", errstr ? ": " : "");
	return (0);
}

void*
kvm_openfiles(const char* execfile, const char* corefile, const char* swapfile, int flags, char* errout)
{
	if (errout) strcpy(errout, "/dev/mem: No such file or directory");
	return (0);
}

int
kvm_read(void* kd, unsigned long addr, void* buf, unsigned int nbytes)
{
	return (-1);
}

int
kvm_uread(void* kd, void* p, unsigned long uva, void* buf, size_t len)
{
	return (0);
}

int
kvm_write(void* kd, unsigned long addr, const void* buf, unsigned int nbytes)
{
	return (0);
}
