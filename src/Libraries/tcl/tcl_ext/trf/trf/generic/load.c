/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 25, 2022.
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
#include "transformInt.h"

#define Offset(type,field) ((unsigned long) (((char *) &((type *) 0)->field)))

/*
 *----------------------------------------------------------------------
 *
 * Trf_LoadLibrary --
 *
 *	This procedure is called to load a shared library into memory.
 *	If the extension is ".so" (e.g. Solaris, Linux) or ".sl" (HP-UX)
 *	it is possible that the extension is appended or replaced with
 *	a major version number. If the file cannot be found, the version
 *	numbers will be stripped off one by one. e.g.
 *
 *	HP-UX:	libtiff.3.4	Linux,Solaris:	libtiff.so.3.4
 *		libtiff.3			libtiff.so.3
 *		libtiff.sl			libtiff.so
 *
 * Results:
 *	TCL_OK if function succeeds. Otherwise TCL_ERROR while the
 *	interpreter will contain an error-message. The last parameter
 *	"num" contains the minimum number of symbols that is required
 *	by the application to succeed. Only the first <num> symbols
 *	will produce an error if they cannot be found.
 *
 * Side effects:
 *	At least <num> Library functions become available by the
 *	application.
 *
 *----------------------------------------------------------------------
 */

typedef struct Functions {
    VOID *handle;
    int (* first) _ANSI_ARGS_((void));
    int (* next) _ANSI_ARGS_((void));
} Functions;

/* MS defines something under this name, avoid the collision
 */

#define TRF_LOAD_FAILED ((VOID *) -114)

int
Trf_LoadLibrary (interp, libName, handlePtr, symbols, num)
    Tcl_Interp *interp;
    CONST char *libName;
    VOID **handlePtr;
    char **symbols;
    int num;
{
    VOID *handle = (VOID *) NULL;
    Functions *lib = (Functions *) handlePtr;
    char **p = (char **) &(lib->first);
    char **q = symbols;
    char buf[256];
    char *r;
    int length;

    if (lib->handle != NULL) {
      if (lib->handle == TRF_LOAD_FAILED) {
	Tcl_AppendResult (interp, "cannot open ", (char*) NULL);
	Tcl_AppendResult (interp, libName, (char*) NULL);
      }
      return (lib->handle != TRF_LOAD_FAILED) ? TCL_OK : TCL_ERROR;
    }

    length = strlen(libName);
    strcpy(buf,libName);
    handle = dlopen(buf, RTLD_NOW);

    while (handle == NULL) {
	if ((r = strrchr(buf,'.')) != NULL) {
	    if ((r[1] < '0') || (r[1] > '9')) {
	        Tcl_AppendResult (interp, "cannot open ", (char*) NULL);
	        Tcl_AppendResult (interp, libName, (char*) NULL);
	        Tcl_AppendResult (interp, ": ", (char*) NULL);
	        Tcl_AppendResult (interp, dlerror (), (char*) NULL);
		lib->handle = TRF_LOAD_FAILED;
		return TCL_ERROR;
	    }
	    length = r - buf;
	    *r = 0;
	}
	if (strchr(buf,'.') == NULL) {
	    strcpy(buf+length,".sl");
	    length += 3;
	}
	dlerror();
	handle = dlopen(buf, RTLD_NOW);
    }

    buf[0] = '_';
    while (*q) {
	*p = (char *) dlsym(handle,*q);
	if (*p == (char *)NULL) {
	    strcpy(buf+1,*q);
	    *p = (char *) dlsym(handle,buf);
	    if ((num > 0) && (*p == (char *)NULL)) {
	        Tcl_AppendResult (interp, "cannot open ", (char*) NULL);
	        Tcl_AppendResult (interp, libName, (char*) NULL);
	        Tcl_AppendResult (interp, ": symbol \"", (char*) NULL);
	        Tcl_AppendResult (interp, *q, (char*) NULL);
	        Tcl_AppendResult (interp, "\" not found", (char*) NULL);
		dlclose(handle);
		lib->handle = TRF_LOAD_FAILED;
		return TCL_ERROR;
	    }
	}
	q++; num--;
	p += (Offset(Functions, next) - Offset(Functions, first)) /
		sizeof(char *);
    }
    lib->handle = handle;

    return TCL_OK;
}

/*
 *----------------------------------------------------------------------
 *
 * Trf_LoadFailed --
 *
 *	Mark the loaded library as invalid. Remove it from memory
 *	if possible. It will no longer be used in the future.
 *
 * Results:
 *	None.
 *
 * Side effects:
 *	Next time the same handle is used by TrfLoadLib, it will
 *	fail immediately, without trying to load it.
 *
 *----------------------------------------------------------------------
 */

void
Trf_LoadFailed (handlePtr)
    VOID **handlePtr;
{
    if ((*handlePtr != NULL) && (*handlePtr != TRF_LOAD_FAILED)) {
	/* Oops, still loaded. First remove it from menory */
	dlclose(*handlePtr);
    }
    *handlePtr = TRF_LOAD_FAILED;
}
