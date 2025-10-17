/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 8, 2021.
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
TclStubs *tclStubsPtr;
TclPlatStubs *tclPlatStubsPtr;
struct TclIntStubs *tclIntStubsPtr;
struct TclIntPlatStubs *tclIntPlatStubsPtr;

static int MyInitStubs(Tcl_Interp *ip) {
  typedef struct  {
    char *result;
    Tcl_FreeProc *freeProc;
    int errorLine;
    TclStubs *stubTable;
  } HeadOfInterp;

  HeadOfInterp *hoi = (HeadOfInterp*)ip;

  if (hoi->stubTable == NULL || hoi->stubTable->magic != TCL_STUB_MAGIC) {
    ip->result = "This extension requires stubs-support.";
    ip->freeProc = TCL_STATIC;
    return 0;
  }

  tclStubsPtr = hoi->stubTable;

  if (Tcl_PkgRequire(ip, "Tcl", "8.1", 0) == NULL) {
    tclStubsPtr = NULL;
    return 0;
  }

  if (tclStubsPtr->hooks != NULL) {
    tclPlatStubsPtr = tclStubsPtr->hooks->tclPlatStubs;
    tclIntStubsPtr = tclStubsPtr->hooks->tclIntStubs;
    tclIntPlatStubsPtr = tclStubsPtr->hooks->tclIntPlatStubs;
  }

  return 1;
}
