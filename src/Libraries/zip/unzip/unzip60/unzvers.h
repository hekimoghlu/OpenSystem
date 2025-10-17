/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 28, 2022.
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
/*
   unzvers.h (for UnZip) by Info-ZIP.
 */

#ifndef __unzvers_h     /* don't include more than once */
#define __unzvers_h

#ifdef BETA
#  undef BETA           /* undefine BETA for public releases */
#endif

#ifdef BETA
#  define UZ_BETALEVEL      "h03 BETA"
#  define UZ_VERSION_DATE   "17 Apr 09"         /* internal beta version */
#else
#  define UZ_BETALEVEL      ""
#  define UZ_VERSION_DATE   "20 April 2009"     /* official release version */
#  define RELEASE
#endif

#define UZ_MAJORVER    6   /* UnZip */
#define UZ_MINORVER    0

#define ZI_MAJORVER    3   /* ZipInfo */
#define ZI_MINORVER    0

#define UZ_PATCHLEVEL  0

#define UZ_VER_STRING  "6.0"          /* keep in sync with Version numbers! */

#ifndef IZ_COMPANY_NAME               /* might be already defined... */
#  define IZ_COMPANY_NAME "Info-ZIP"
#endif

/* these are obsolete but remain for backward compatibility: */
#if (defined(OS2) || defined(__OS2__))
#  define D2_MAJORVER    UZ_MAJORVER    /* DLL for OS/2 */
#  define D2_MINORVER    UZ_MINORVER
#  define D2_PATCHLEVEL  UZ_PATCHLEVEL
#endif

#define DW_MAJORVER    UZ_MAJORVER    /* DLL for MS Windows */
#define DW_MINORVER    UZ_MINORVER
#define DW_PATCHLEVEL  UZ_PATCHLEVEL

#define WIN_VERSION_DATE  UZ_VERSION_DATE

#define UNZ_DLL_VERSION   UZ_VER_STRING

/* The following version constants specify the UnZip version that introduced
 * the most recent incompatible change (means: change that breaks backward
 * compatibility) of a DLL/Library binary API definition.
 *
 * Currently, UnZip supports three distinct DLL/Library APIs, which each
 * carry their own "compatibility level":
 * a) The "generic" (console-mode oriented) API has been used on UNIX,
 *    for example. This API provides a "callable" interface similar to the
 *    interactive command line of the normal program executables.
 * b) The OS/2-only API provides (additional) functions specially tailored
 *    for interfacing with the REXX shell.
 * c) The Win32 DLL API with a pure binary interface which can be used to
 *    build GUI mode as well as Console mode applications.
 *
 * Whenever a change that breaks backward compatibility gets applied to
 * any of the DLL/Library APIs, the corresponding compatibility level should
 * be synchronized with the current UnZip version numbers.
 */
/* generic DLL API minimum compatible version*/
#define UZ_GENAPI_COMP_MAJOR  6
#define UZ_GENAPI_COMP_MINOR  0
#define UZ_GENAPI_COMP_REVIS  0
/* os2dll API minimum compatible version*/
#define UZ_OS2API_COMP_MAJOR  6
#define UZ_OS2API_COMP_MINOR  0
#define UZ_OS2API_COMP_REVIS  0
/* windll API minimum compatible version*/
#define UZ_WINAPI_COMP_MAJOR  6
#define UZ_WINAPI_COMP_MINOR  0
#define UZ_WINAPI_COMP_REVIS  0

#endif /* !__unzvers_h */
