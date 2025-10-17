/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 20, 2025.
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
#include <assert.h>
#include <stdlib.h>
#include <string.h>

#if BC_ENABLE_NLS
#include <locale.h>
#endif // BC_ENABLE_NLS

#ifndef _WIN32
#include <libgen.h>
#endif // _WIN32

#include <setjmp.h>

#include <version.h>
#include <status.h>
#include <vm.h>
#include <bc.h>
#include <dc.h>

int
main(int argc, char* argv[])
{
	BcStatus s;
	char* name;
	size_t len = strlen(BC_EXECPREFIX);

#if BC_ENABLE_NLS
	// Must set the locale properly in order to have the right error messages.
	vm->locale = setlocale(LC_ALL, "");
#endif // BC_ENABLE_NLS

	// Set the start pledge().
	bc_pledge(bc_pledge_start, NULL);

	// Sometimes, argv[0] can be NULL. Better make sure to be robust against it.
	if (argv[0] != NULL)
	{
		// Figure out the name of the calculator we are using. We can't use
		// basename because it's not portable, but yes, this is stripping off
		// the directory.
		name = strrchr(argv[0], BC_FILE_SEP);
		vm->name = (name == NULL) ? argv[0] : name + 1;
	}
	else
	{
#if !DC_ENABLED
		vm->name = "bc";
#elif !BC_ENABLED
		vm->name = "dc";
#else
		// Just default to bc in that case.
		vm->name = "bc";
#endif
	}

	// If the name is longer than the length of the prefix, skip the prefix.
	if (strlen(vm->name) > len) vm->name += len;

	BC_SIG_LOCK;

	// We *must* do this here. Otherwise, other code could not jump out all of
	// the way.
	bc_vec_init(&vm->jmp_bufs, sizeof(sigjmp_buf), BC_DTOR_NONE);

	BC_SETJMP_LOCKED(vm, exit);

#if !DC_ENABLED
	s = bc_main(argc, argv);
#elif !BC_ENABLED
	s = dc_main(argc, argv);
#else
	// BC_IS_BC uses vm->name, which was set above. So we're good.
	if (BC_IS_BC) s = bc_main(argc, argv);
	else s = dc_main(argc, argv);
#endif

	vm->status = (int) s;

exit:
	BC_SIG_MAYLOCK;

	return vm->status == BC_STATUS_QUIT ? BC_STATUS_SUCCESS : vm->status;
}
