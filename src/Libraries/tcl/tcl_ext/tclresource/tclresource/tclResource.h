/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 11, 2025.
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

// -------------------------------------------------------
// File: "tclResource.h"
//                        Created: 2003-09-20 10:13:07
//              Last modification: 2005-12-27 17:12:12
// Author: Bernard Desgraupes
// e-mail: <bdesgraupes@users.sourceforge.net>
// www: <http://sourceforge.net/projects/tclresource>
// (c) Copyright : Bernard Desgraupes, 2003-2004, 2005
// All rights reserved.
// -------------------------------------------------------

#ifndef _TCLRESOURCE_H
#define _TCLRESOURCE_H

#define TARGET_API_MAC_CARBON	1
#define TARGET_API_MAC_OSX		1

// Stubs mechanism enabled
#define USE_TCL_STUBS

#include <Carbon/Carbon.h>

struct SFReply {char dummy;};
typedef struct SFReply SFReply;
typedef struct SFReply StandardFileReply;


#if TARGET_RT_MAC_MACHO
	#ifdef MAC_TCL
		#undef MAC_TCL
	#endif
#endif

#include <Tcl/tcl.h>
#include <Tcl/tclInt.h>

#ifndef CONST84 // Tcl 8.4 backwards compatibility
#      define CONST84 
#      define CONST84_RETURN CONST
#endif


#endif // _TCLRESOURCE_H
