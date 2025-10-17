/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 13, 2025.
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
#include <memchan.h>

#undef TCL_STORAGE_CLASS
#define TCL_STORAGE_CLASS DLLEXPORT

int TestObjCmd(ClientData clientData, Tcl_Interp *interp, 
               int objc, Tcl_Obj *const objv[]);


EXTERN int
Chantest_Init(Tcl_Interp *interp)
{
    int r = TCL_OK;
    const char *tcl = NULL;
    const char *memchan = NULL;

#ifdef USE_TCL_STUBS
    tcl = Tcl_InitStubs(interp, "8.4", 0);
#endif

#ifdef USE_MEMCHAN_STUBS
    memchan = Memchan_InitStubs(interp, "2.2", 0);
#endif

    if (tcl == NULL || memchan == NULL) {
        Tcl_SetResult(interp, "error loading memchan via stubs", TCL_STATIC);
        r = TCL_ERROR;
    } else {
        Tcl_CreateObjCommand(interp, "chantest", TestObjCmd, 
                             (ClientData)NULL, (Tcl_CmdDeleteProc *)NULL);
        r = Tcl_PkgProvide(interp, "Chantest", "0.1");
    }
    return r;
}

int
TestObjCmd(ClientData clientData, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[])
{
    Tcl_Channel chan = NULL, chan2 = NULL;
    Tcl_Obj *resObj = NULL;
    char *type = "";
    int r = TCL_OK;

    if (objc != 2) {
        Tcl_WrongNumArgs(interp, 1, objv, "type");
        return TCL_ERROR;
    }
    
    type = Tcl_GetString(objv[1]);
    if (strcmp("memchan", type) == 0) {
        chan = Memchan_CreateMemoryChannel(interp, 0);
    } else if (strcmp("fifo", type) == 0) {
        chan = Memchan_CreateFifoChannel(interp);
    } else if (strcmp("fifo2", type) == 0) {
        Memchan_CreateFifo2Channel(interp, &chan, &chan2);
    } else if (strcmp("null", type) == 0) {
        chan = Memchan_CreateNullChannel(interp);
    } else if (strcmp("zero", type) == 0) {
        chan = Memchan_CreateZeroChannel(interp);
    } else if (strcmp("random", type) == 0) {
        chan = Memchan_CreateRandomChannel(interp);
    }
    
    if (chan2 != NULL) {
        Tcl_Obj *name[2];
        name[0] = Tcl_NewStringObj(Tcl_GetChannelName(chan), -1);
        name[1] = Tcl_NewStringObj(Tcl_GetChannelName(chan2), -1);
        resObj = Tcl_NewListObj(2, name);
        r = TCL_OK;
    } else if (chan != NULL) {
        resObj = Tcl_NewStringObj(Tcl_GetChannelName(chan), -1);
        r = TCL_OK;
    } else {
        resObj = Tcl_NewStringObj("error", -1);
        r = TCL_ERROR;
    }
    
    Tcl_SetObjResult(interp, resObj);
    return r;
}
