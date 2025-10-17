/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 5, 2022.
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
#ifdef NS_AOLSERVER
#include <ns.h>

int Ns_ModuleVersion = 1;

/*
 * Structure to pass to NsThread_Init. This holds the module
 * and virtual server name for proper interp initializations. 
 */

struct mydata {
    char *modname;
    char *server;
};

/*
 *----------------------------------------------------------------------------
 *
 * NsThread_Init --
 *
 *    Loads the package for the first time, i.e. in the startup thread.
 *
 * Results:
 *    Standard Tcl result
 *
 * Side effects:
 *    Package initialized. Tcl commands created.
 *
 *----------------------------------------------------------------------------
 */

static int
NsThread_Init (Tcl_Interp *interp, void *cd)
{
    struct mydata *md = (struct mydata*)cd;
    int ret = Thread_Init(interp);

    if (ret != TCL_OK) {
        Ns_Log(Warning, "can't load module %s: %s", md->modname,
               Tcl_GetStringResult(interp));
        return TCL_ERROR;
    }
    Tcl_SetAssocData(interp, "thread:nsd", NULL, (ClientData)md);

    return TCL_OK;
}

/*
 *----------------------------------------------------------------------------
 *
 * Ns_ModuleInit --
 *
 *    Called by the AOLserver when loading shared object file.
 *
 * Results:
 *    Standard AOLserver result
 *
 * Side effects:
 *    Many. Depends on the package.
 *
 *----------------------------------------------------------------------------
 */

int
Ns_ModuleInit(char *srv, char *mod)
{
    struct mydata *md = NULL;

    md = (struct mydata*)ns_malloc(sizeof(struct mydata));
    md->modname = strcpy(ns_malloc(strlen(mod)+1), mod);
    md->server  = strcpy(ns_malloc(strlen(srv)+1), srv);

    return (Ns_TclInitInterps(srv, NsThread_Init, (void*)md) == TCL_OK)
        ? NS_OK : NS_ERROR; 
}

#endif /* NS_AOLSERVER */

/* EOF $RCSfile: aolstub.cpp,v $ */

/* Emacs Setup Variables */
/* Local Variables:      */
/* mode: C               */
/* indent-tabs-mode: nil */
/* c-basic-offset: 4     */
/* End:                  */
