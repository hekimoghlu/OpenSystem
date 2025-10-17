/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 6, 2022.
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

// mwcw.h --
// $Id: mwcw.h 1230 2007-03-09 15:58:53Z jcw $
// This is part of Metakit, see http://www.equi4.com/metakit.html

/** @file
 * Configuration header for Metrowerks CodeWarrior
 */

#define q4_MWCW 1

/////////////////////////////////////////////////////////////////////////////

#if q4_68K
#if !__option(IEEEdoubles)
#error Cannot build Metakit with 10-byte doubles
#endif 
#endif 

#if __option(bool)
#define q4_BOOL 1
// undo previous defaults, because q4_BOOL is not set early enough
#undef false
#undef true
#undef bool
#endif 

#undef _MSC_VER

#pragma export on

/////////////////////////////////////////////////////////////////////////////
