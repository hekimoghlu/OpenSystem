/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 27, 2021.
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

// win.h --
// $Id: win.h 1230 2007-03-09 15:58:53Z jcw $
// This is part of Metakit, the homepage is http://www.equi4.com/metakit.html

/** @file
 * Configuration header for Windows builds
 */

#if defined (_MSDOS)
#define q4_DOS 1
#endif 

#if defined (_WINDOWS)
#define q4_WIN 1
#endif 

#if defined (_WIN32)
#define q4_WIN32 1
#endif 

#if defined (_WIN32_WCE)	// check for Win CE
#define q4_WINCE 1
#define q4_WIN32 1
#endif 

#if q4_WIN32                    // WIN32 implies WIN
#undef q4_WIN
#define q4_WIN 1
#endif 

#if q4_WIN                      // WIN implies not DOS, even for Win3
#undef q4_DOS
#endif
