/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 5, 2024.
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

/*
 * ast dynamic data initialization
 */

#ifdef _UWIN

#define _std_def_cfree	1

#include <sfio_t.h>
#include <ast.h>

#undef	strcoll

#include <ast_windows.h>

extern Sfio_t	_Sfstdin;
extern Sfio_t	_Sfstdout;
extern Sfio_t	_Sfstderr;

#include "sfhdr.h"

#undef	sfstdin
#undef	sfstdout
#undef	sfstderr

#if defined(__EXPORT__)
#define extern		__EXPORT__
#endif

/*
 * for backward compatibility with early UNIX
 */

extern void
cfree(void* addr)
{
	free(addr);
}

extern void
_ast_libinit(void* in, void* out, void* err)
{
	Sfio_t*		sp;

	sp = (Sfio_t*)in;
	*sp =  _Sfstdin;
	sfstdin = sp;
	sp = (Sfio_t*)out;
	*sp =  _Sfstdout;
	sfstdout = sp;
	sp = (Sfio_t*)err;
	*sp =  _Sfstderr;
	sfstderr = sp;
}

extern void
_ast_init(void)
{
	struct _astdll*	ap = _ast_getdll();

	_ast_libinit(ap->_ast_stdin,ap->_ast_stdout,ap->_ast_stderr);
}

extern void
_ast_exit(void)
{
	if (_Sfcleanup)
		(*_Sfcleanup)();
}

BOOL WINAPI
DllMain(HINSTANCE hinst, DWORD reason, VOID* reserved)
{
	switch (reason)
	{
	case DLL_PROCESS_ATTACH:
		break;
	case DLL_PROCESS_DETACH:
		_ast_exit();
		break;
	}
	return 1;
}

#else

#include <ast.h>

#if _dll_data_intercept && ( _DLL_BLD || _BLD_DLL )

#undef	environ

extern char**	environ;

struct _astdll	_ast_dll = { &environ };

struct _astdll*
_ast_getdll(void)
{
	return &_ast_dll;
}

#else

NoN(astdynamic)

#endif

#endif
