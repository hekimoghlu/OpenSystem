/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 1, 2023.
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

// mfc.h --
// $Id: mfc.h 1230 2007-03-09 15:58:53Z jcw $
// This is part of Metakit, see http://www.equi4.com/metakit.html

/** @file
 * Configuration header for MFC-based builds
 */

#define q4_MFC 1

#if q4_WIN && !q4_WIN32
#include <afxwin.h>
#else 
#include <afxcoll.h>
#endif 

#undef d4_assert
#define d4_assert ASSERT

#undef d4_assertThis
#define d4_assertThis d4_assert(AfxIsValidAddress(this, sizeof *this, FALSE))

#undef d4_new
#define d4_new DEBUG_NEW

typedef class CString c4_String;
typedef class CPtrArray c4_PtrArray;
typedef class CDWordArray c4_DWordArray;
typedef class CStringArray c4_StringArray;

// MSVC 1.52 thinks a typedef has no constructor, so use a define instead
#if !q4_OK && q4_MSVC && _MSC_VER == 800
#define c4_String CString
#endif
