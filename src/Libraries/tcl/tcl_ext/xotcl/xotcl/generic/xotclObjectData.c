/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 6, 2024.
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
#include "xotclInt.h"

#ifdef XOTCL_OBJECTDATA
extern void
XOTclFreeObjectData(XOTclClass* cl) {
  if (cl->opt && cl->opt->objectdata) {
    Tcl_DeleteHashTable(cl->opt->objectdata);
    ckfree((char*)cl->opt->objectdata);
    cl->opt->objectdata = 0; 
  }
}
extern void
XOTclSetObjectData(XOTclObject* obj, XOTclClass* cl, ClientData data) {
  Tcl_HashEntry *hPtr;
  int nw;

  XOTclRequireClassOpt(cl);

  if (!cl->opt->objectdata) {
    cl->opt->objectdata = (Tcl_HashTable*)ckalloc(sizeof(Tcl_HashTable));
    Tcl_InitHashTable(cl->opt->objectdata, TCL_ONE_WORD_KEYS);
  }
  hPtr = Tcl_CreateHashEntry(cl->opt->objectdata, (char*)obj, &nw);
  Tcl_SetHashValue(hPtr, data);
}

extern int
XOTclGetObjectData(XOTclObject* obj, XOTclClass* cl, ClientData* data) {
  Tcl_HashEntry *hPtr;
  if (!cl->opt || !cl->opt->objectdata) 
    return 0;
  hPtr = Tcl_FindHashEntry(cl->opt->objectdata, (char*)obj);
  if (data) *data = hPtr ? Tcl_GetHashValue(hPtr) : 0;
  return hPtr != 0;
}

extern int
XOTclUnsetObjectData(XOTclObject* obj, XOTclClass* cl) {
  Tcl_HashEntry *hPtr;

  if (!cl->opt || !cl->opt->objectdata) 
    return 0;
  hPtr = Tcl_FindHashEntry(cl->opt->objectdata, (char*)obj);
  if (hPtr) Tcl_DeleteHashEntry(hPtr);
  return hPtr != 0;
}
#endif
