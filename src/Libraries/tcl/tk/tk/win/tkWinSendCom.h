/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 14, 2023.
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
#ifndef _tkWinSendCom_h_INCLUDE
#define _tkWinSendCom_h_INCLUDE

#include "tkWinInt.h"
#include <ole2.h>

#ifdef _MSC_VER
#   pragma comment (lib, "ole32.lib")
#   pragma comment (lib, "oleaut32.lib")
#   pragma comment (lib, "uuid.lib")
#endif

/*
 * TkWinSendCom CoClass structure
 */

typedef struct {
    IDispatchVtbl *lpVtbl;
    ISupportErrorInfoVtbl *lpVtbl2;
    long refcount;
    Tcl_Interp *interp;
} TkWinSendCom;

/*
 * TkWinSendCom Dispatch IDs
 */

#define TKWINSENDCOM_DISPID_SEND   1
#define TKWINSENDCOM_DISPID_ASYNC  2

/*
 * TkWinSendCom public functions
 */

HRESULT                 TkWinSendCom_CreateInstance(Tcl_Interp *interp,
                            REFIID riid, void **ppv);
int                     TkWinSend_QueueCommand(Tcl_Interp *interp,
                            Tcl_Obj *cmdPtr);
void                    SetExcepInfo(Tcl_Interp *interp,
                            EXCEPINFO *pExcepInfo);

#endif /* _tkWinSendCom_h_INCLUDE */

/*
 * Local Variables:
 * mode: c
 * End:
 */
