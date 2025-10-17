/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 28, 2025.
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
#ifndef _TCLPLATDECLS
#define _TCLPLATDECLS

#undef TCL_STORAGE_CLASS
#ifdef BUILD_tcl
#   define TCL_STORAGE_CLASS DLLEXPORT
#else
#   ifdef USE_TCL_STUBS
#      define TCL_STORAGE_CLASS
#   else
#      define TCL_STORAGE_CLASS DLLIMPORT
#   endif
#endif

/*
 *  Pull in the typedef of TCHAR for windows.
 */
#if defined(__WIN32__) && !defined(_TCHAR_DEFINED)
#   include <tchar.h>
#   ifndef _TCHAR_DEFINED
	/* Borland seems to forget to set this. */
	typedef _TCHAR TCHAR;
#	define _TCHAR_DEFINED
#   endif
#endif

/* !BEGIN!: Do not edit below this line. */

/*
 * Exported function declarations:
 */

#ifdef __WIN32__ /* WIN */
#ifndef Tcl_WinUtfToTChar_TCL_DECLARED
#define Tcl_WinUtfToTChar_TCL_DECLARED
/* 0 */
EXTERN TCHAR *		Tcl_WinUtfToTChar(CONST char *str, int len,
				Tcl_DString *dsPtr);
#endif
#ifndef Tcl_WinTCharToUtf_TCL_DECLARED
#define Tcl_WinTCharToUtf_TCL_DECLARED
/* 1 */
EXTERN char *		Tcl_WinTCharToUtf(CONST TCHAR *str, int len,
				Tcl_DString *dsPtr);
#endif
#endif /* WIN */
#ifdef MAC_OSX_TCL /* MACOSX */
#ifndef Tcl_MacOSXOpenBundleResources_TCL_DECLARED
#define Tcl_MacOSXOpenBundleResources_TCL_DECLARED
/* 0 */
EXTERN int		Tcl_MacOSXOpenBundleResources(Tcl_Interp *interp,
				CONST char *bundleName, int hasResourceFile,
				int maxPathLen, char *libraryPath);
#endif
#ifndef Tcl_MacOSXOpenVersionedBundleResources_TCL_DECLARED
#define Tcl_MacOSXOpenVersionedBundleResources_TCL_DECLARED
/* 1 */
EXTERN int		Tcl_MacOSXOpenVersionedBundleResources(
				Tcl_Interp *interp, CONST char *bundleName,
				CONST char *bundleVersion,
				int hasResourceFile, int maxPathLen,
				char *libraryPath);
#endif
#endif /* MACOSX */

typedef struct TclPlatStubs {
    int magic;
    struct TclPlatStubHooks *hooks;

#ifdef __WIN32__ /* WIN */
    TCHAR * (*tcl_WinUtfToTChar) (CONST char *str, int len, Tcl_DString *dsPtr); /* 0 */
    char * (*tcl_WinTCharToUtf) (CONST TCHAR *str, int len, Tcl_DString *dsPtr); /* 1 */
#endif /* WIN */
#ifdef MAC_OSX_TCL /* MACOSX */
    int (*tcl_MacOSXOpenBundleResources) (Tcl_Interp *interp, CONST char *bundleName, int hasResourceFile, int maxPathLen, char *libraryPath); /* 0 */
    int (*tcl_MacOSXOpenVersionedBundleResources) (Tcl_Interp *interp, CONST char *bundleName, CONST char *bundleVersion, int hasResourceFile, int maxPathLen, char *libraryPath); /* 1 */
#endif /* MACOSX */
} TclPlatStubs;

#ifdef __cplusplus
extern "C" {
#endif
extern TclPlatStubs *tclPlatStubsPtr;
#ifdef __cplusplus
}
#endif

#if defined(USE_TCL_STUBS) && !defined(USE_TCL_STUB_PROCS)

/*
 * Inline function declarations:
 */

#ifdef __WIN32__ /* WIN */
#ifndef Tcl_WinUtfToTChar
#define Tcl_WinUtfToTChar \
	(tclPlatStubsPtr->tcl_WinUtfToTChar) /* 0 */
#endif
#ifndef Tcl_WinTCharToUtf
#define Tcl_WinTCharToUtf \
	(tclPlatStubsPtr->tcl_WinTCharToUtf) /* 1 */
#endif
#endif /* WIN */
#ifdef MAC_OSX_TCL /* MACOSX */
#ifndef Tcl_MacOSXOpenBundleResources
#define Tcl_MacOSXOpenBundleResources \
	(tclPlatStubsPtr->tcl_MacOSXOpenBundleResources) /* 0 */
#endif
#ifndef Tcl_MacOSXOpenVersionedBundleResources
#define Tcl_MacOSXOpenVersionedBundleResources \
	(tclPlatStubsPtr->tcl_MacOSXOpenVersionedBundleResources) /* 1 */
#endif
#endif /* MACOSX */

#endif /* defined(USE_TCL_STUBS) && !defined(USE_TCL_STUB_PROCS) */

/* !END!: Do not edit above this line. */

#undef TCL_STORAGE_CLASS
#define TCL_STORAGE_CLASS DLLIMPORT

#endif /* _TCLPLATDECLS */


