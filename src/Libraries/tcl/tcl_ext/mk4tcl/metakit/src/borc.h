/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 12, 2024.
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

// borc.h --
// $Id: borc.h 1230 2007-03-09 15:58:53Z jcw $
// This is part of Metakit, see http://www.equi4.com/metakit.html

/** @file
 * Configuration header for Borland C++
 */

#define q4_BORC 1

// get rid of several common warning messages
#if !q4_STRICT
#pragma warn -aus // 'identifier' is assigned a value that is never used
#pragma warn -par // Parameter 'parameter' is never used.
#pragma warn -sig // Conversion may lose significant digits
#pragma warn -use // 'identifier' declared but never used
#endif 

#if __BORLANDC__ >= 0x500
#define q4_BOOL 1     // supports the bool datatype
// undo previous defaults, because q4_BOOL is not set early enough
#undef false
#undef true
#undef bool
#endif 

#if !defined (q4_EXPORT)
#define q4_EXPORT 1     // requires export/import specifiers
#endif 

#if defined (__MT__)
#define q4_MULTI 1      // uses multi-threading
#endif
