/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 28, 2025.
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
#pragma prototyped

#include	<shell.h>

#include	"FEATURE/externs"

#if defined(__sun) && _sys_mman && _lib_memcntl && defined(MHA_MAPSIZE_STACK) && defined(MC_HAT_ADVISE)
#   undef	VM_FLAGS	/* solaris vs vmalloc.h symbol clash */
#   include	<sys/mman.h>
#else
#   undef	_lib_memcntl
#endif

typedef int (*Shnote_f)(int, long, int);

int main(int argc, char *argv[])
{
#if _lib_memcntl
	/* advise larger stack size */
	struct memcntl_mha mha;
	mha.mha_cmd = MHA_MAPSIZE_STACK;
	mha.mha_flags = 0;
	mha.mha_pagesize = 64 * 1024;
	(void)memcntl(NULL, 0, MC_HAT_ADVISE, (caddr_t)&mha, 0, 0);
#endif
	return(sh_main(argc, argv, (Shinit_f)0));
}
