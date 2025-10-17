/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 25, 2022.
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

#pragma once on#include "CoreHeadersMach-O.h"// These symbols are defined from MSL MacHeadersMach-O.h // (errno.h and stat.h are in the Kernel.framework)// and are redefined later in TclErrno.h : undef them// to avoid error message#undef	EOVERFLOW#undef	EOPNOTSUPP// This avoids the loading of stat.h from tclMacPort.h#define	_MSL_STAT_H// ---------------------------------------------------------------// Replace #include "tclMacCommonPch.h" by its partial contents.#if !__option(enumsalwaysint)#error Tcl requires the Metrowerks setting "Enums always ints".#endif// Tell Tcl (or any Tcl extensions) that we are compiling for the Macintosh platform.#define MAC_TCL// ---------------------------------------------------------------#define USE_TCL_STUBS 1// See dom.h for this one:#define USE_NORMAL_ALLOCATOR#define TCL_MEM_DEBUG#define MAC_OSX_TCL#define TDOM_NO_UNKNOWN_CMD#define VERSION "0.8.3"#include <Tcl/tcl.h>
