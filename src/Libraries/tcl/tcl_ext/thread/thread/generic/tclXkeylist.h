/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 3, 2024.
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
#ifndef _KEYLIST_H_
#define _KEYLIST_H_

/* 
 * Keyed list object interface commands
 */

Tcl_Obj* TclX_NewKeyedListObj();

void TclX_KeyedListInit(Tcl_Interp*);
int  TclX_KeyedListGet(Tcl_Interp*, Tcl_Obj*, const char*, Tcl_Obj**);
int  TclX_KeyedListSet(Tcl_Interp*, Tcl_Obj*, const char*, Tcl_Obj*);
int  TclX_KeyedListDelete(Tcl_Interp*, Tcl_Obj*, const char*);
int  TclX_KeyedListGetKeys(Tcl_Interp*, Tcl_Obj*, const char*, Tcl_Obj**);

/*
 * Exported for usage in Sv_DuplicateObj. This is slightly
 * modified version of the DupKeyedListInternalRep() function.
 * It does a proper deep-copy of the keyed list object.
 */

void DupKeyedListInternalRepShared(Tcl_Obj*, Tcl_Obj*);

#endif /* _KEYLIST_H_ */

/* EOF $RCSfile: tclXkeylist.h,v $ */

/* Emacs Setup Variables */
/* Local Variables:      */
/* mode: C               */
/* indent-tabs-mode: nil */
/* c-basic-offset: 4     */
/* End:                  */

