/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 16, 2025.
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
#ifndef ASSERT_VERIFY_H
#define ASSERT_VERIFY_H

/******************************************************************************
Written by: Jeffrey Richter
Notices: Copyright (c) 1995-1997 Jeffrey Richter
Purpose: Common header file containing handy macros and definitions used
         throughout all the applications in the book.
******************************************************************************/

/* These header functions were copied from the cmnhdr.h file that accompanies 
   Advanced Windows 3rd Edition by Jeffrey Richter */

//////////////////////////// Assert/Verify Macros /////////////////////////////

#if		defined(macintosh) || defined(__APPLE__)
/* TBD */
#define chFAIL(szMSG)                                          
#define chASSERTFAIL(file,line,expr) 
#else
#define chFAIL(szMSG) {                                                   \
      MessageBox(GetActiveWindow(), szMSG,                                \
         __TEXT("Assertion Failed"), MB_OK | MB_ICONERROR);               \
      DebugBreak();                                                       \
   }

/* Put up an assertion failure message box. */
#define chASSERTFAIL(file,line,expr) {                                    \
      TCHAR sz[128];                                                      \
      wsprintf(sz, __TEXT("File %hs, line %d : %hs"), file, line, expr);  \
      chFAIL(sz);                                                         \
   }

#endif	/* macintosh */

/* Put up a message box if an assertion fails in a debug build. */
#ifdef _DEBUG
#define chASSERT(x) if (!(x)) chASSERTFAIL(__FILE__, __LINE__, #x)
#else
#define chASSERT(x)
#endif

/* Assert in debug builds, but don't remove the code in retail builds. */
#ifdef _DEBUG
#define chVERIFY(x) chASSERT(x)
#else
#define chVERIFY(x) (x)
#endif

#endif	/* ASSERT_VERIFY_H */
