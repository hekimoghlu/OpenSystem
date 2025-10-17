/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 4, 2024.
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

// msvc.h --
// $Id: msvc.h 1230 2007-03-09 15:58:53Z jcw $
// This is part of Metakit, the homepage is http://www.equi4.com/metakit.html

/** @file
 * Configuration header for Microsoft Visual C++
 */

#define q4_MSVC 1

// get rid of several common warning messages
#if !q4_STRICT
//#pragma warning(disable: 4244) // conversion ..., possible loss of data
//#pragma warning(disable: 4135) // conversion between diff. integral types
//#pragma warning(disable: 4511) // copy constructor could not be generated
//#pragma warning(disable: 4512) // assignment op could not be generated
//#pragma warning(disable: 4514) // unreferenced inline removed
#pragma warning(disable: 4710) // function ... not inlined
#pragma warning(disable: 4711) // ... selected for automatic inline expansion
#pragma warning(disable: 4100) // unreferenced formal parameter
#endif 

#if _MSC_VER >= 1100
#define q4_BOOL 1     // 5.0 supports the bool datatype
#else 
#define q4_NO_NS 1      // 4.x doesn't use namespaces for STL
#endif 

#if defined (_MT)
#define q4_MULTI 1      // uses multi-threading
#endif 

#if defined (_DEBUG) && !defined (q4_CHECK) // use assertions in debug build
#define q4_CHECK 1
#endif 

#if !q4_STD && !q4_UNIV && !defined (q4_MFC)
#define d4_FW_H "mfc.h"   // default for MSVC is to use MFC
#endif
