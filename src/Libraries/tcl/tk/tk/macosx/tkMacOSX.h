/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 27, 2023.
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
#ifndef _TKMAC
#define _TKMAC

#ifndef _TK
#include "tk.h"
#endif

/*
 * Structures and function types for handling Netscape-type in process
 * embedding where Tk does not control the top-level
 */

typedef int (Tk_MacOSXEmbedRegisterWinProc) (long winID, Tk_Window window);
typedef void* (Tk_MacOSXEmbedGetGrafPortProc) (Tk_Window window);
typedef int (Tk_MacOSXEmbedMakeContainerExistProc) (Tk_Window window);
typedef void (Tk_MacOSXEmbedGetClipProc) (Tk_Window window, TkRegion rgn);
typedef void (Tk_MacOSXEmbedGetOffsetInParentProc) (Tk_Window window, void *ulCorner);

#include "tkPlatDecls.h"

#endif /* _TKMAC */
