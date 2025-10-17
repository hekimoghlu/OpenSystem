/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 18, 2025.
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
#ifdef _WIN32
#ifndef _QUICKTIMETCLWIN_H
#define _QUICKTIMETCLWIN_H 1

#	define USE_NON_CONST
#	define WIN32_LEAN_AND_MEAN

/*
 * Windows specific stuff.
 */

#   include <windows.h>
#   include <Stdlib.h>

/*
 * QuickTime stuff.
 */

#   include <QTML.h>
#   include <Movies.h>
#   include <MoviesFormat.h>
#   include <QuickTimeComponents.h>
#   include <QuickTimeVR.h>
#   include <QuickTimeVRFormat.h>
//#   include <FileTypesAndCreators.h>
#   include <MediaHandlers.h>
#   include <ImageCodec.h>

/*
 * Other Mac specific stuff ported(?) to Windows.
 */
 
#   include <Strings.h>
#   include <Gestalt.h>
#   include <FixMath.h>
#   include <Scrap.h>

// Workaround for name clash of QT and X11 symbol
#define Status X11Status

/*
 * Windows specific Tcl/Tk stuff.
 */

#   include "TkWinInt.h"

#undef Status

#endif      /* end of _QUICKTIMETCLWIN_H */
#endif      /* end of _WIN32 */
