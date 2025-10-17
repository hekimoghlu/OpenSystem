/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 7, 2025.
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
 * Windows GUI/Console: main program (EXE) entry point:
 *
 * Ron Aaron <ronaharon@yahoo.com> wrote this and the DLL support code.
 * Adapted by Ken Takata.
 */
#include "vim.h"

// cproto doesn't create a prototype for VimMain()
#ifdef VIMDLL
__declspec(dllimport)
#endif
int VimMain(int argc, char **argv);

#ifdef VIMDLL
# define SaveInst(hInst)    // Do nothing
#else
void SaveInst(HINSTANCE hInst);
#endif

#ifdef FEAT_GUI
    int WINAPI
wWinMain(
    HINSTANCE	hInstance,
    HINSTANCE	hPrevInst UNUSED,
    LPWSTR	lpszCmdLine UNUSED,
    int		nCmdShow UNUSED)
{
    SaveInst(hInstance);
    return VimMain(0, NULL);
}
#else
    int
wmain(int argc UNUSED, wchar_t **argv UNUSED)
{
    SaveInst(GetModuleHandleW(NULL));
    return VimMain(0, NULL);
}
#endif

#ifdef USE_OWNSTARTUP
// Use our own entry point and don't use the default CRT startup code to
// reduce the size of (g)vim.exe.  This works only when VIMDLL is defined.
//
// For MSVC, the /GS- compiler option is needed to avoid the undefined symbol
// error.  (It disables the security check. However, it affects only this
// function and doesn't have any effect on Vim itself.)
// For MinGW, the -nostdlib compiler option and the --entry linker option are
// needed.
# ifdef FEAT_GUI
    void WINAPI
wWinMainCRTStartup(void)
{
    VimMain(0, NULL);
}
# else
    void
wmainCRTStartup(void)
{
    VimMain(0, NULL);
}
# endif
#endif	// USE_OWNSTARTUP


#if defined(VIMDLL) && defined(FEAT_MZSCHEME)

# if defined(_MSC_VER)
static __declspec(thread) void *tls_space;
extern intptr_t _tls_index;
# elif defined(__MINGW32__)
static __thread void *tls_space;
extern intptr_t _tls_index;
# endif

// Get TLS information that is needed for if_mzsch.
    __declspec(dllexport) void
get_tls_info(void ***ptls_space, intptr_t *ptls_index)
{
    *ptls_space = &tls_space;
    *ptls_index = _tls_index;
    return;
}
#endif
