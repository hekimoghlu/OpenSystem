/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 19, 2024.
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
 *   Subtle combination of files and libraries make up the C++ runtime system for kernel modules.  We are dependant on the KernelModule kmod.make and CreateKModInfo.perl scripts to be exactly instep with both this library module and the libkmod module as well.
 *
 *   If you do any maintenance on any of the following files make sure great care is taken to keep them in Sync.
 *   KernelModule.bproj/kmod.make
 *   KernelModule.bproj/CreateKModInfo.perl
 *   KernelModule.bproj/kmodc++/pure.c
 *   KernelModule.bproj/kmodc++/cplus_start.c
 *   KernelModule.bproj/kmodc++/cplus_start.c
 *   KernelModule.bproj/kmodc/c_start.c
 *   KernelModule.bproj/kmodc/c_stop.c
 *
 *   The trick is that the linkline links all of the developers modules.  If any static constructors are used .constructors_used will be left as an undefined symbol.  This symbol is exported by the cplus_start.c routine which automatically brings in the appropriate C++ _start routine.  However the actual _start symbol is only required by the kmod_info structure that is created and initialized by the CreateKModInfo.perl script.  If no C++ was used the _start will be an undefined symbol that is finally satisfied by the c_start module in the kmod library.
 *
 *   The linkline must look like this.
 *.o -lkmodc++ kmod_info.o -lkmod
 */
#include <mach/mach_types.h>

// These global symbols will be defined by CreateInfo script's info.c file.
extern kmod_stop_func_t *_antimain;

__private_extern__ kern_return_t
_stop(kmod_info_t *ki, void *data)
{
	if (_antimain) {
		return (*_antimain)(ki, data);
	} else {
		return KERN_SUCCESS;
	}
}
