/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 22, 2023.
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
#ifndef __XML_EXPORTS_H__
#define __XML_EXPORTS_H__

#if defined(_WIN32) || defined(__CYGWIN__)
/** DOC_DISABLE */

#ifdef LIBXML_STATIC
  #define XMLPUBLIC
#elif defined(IN_LIBXML)
  #define XMLPUBLIC __declspec(dllexport)
#else
  #define XMLPUBLIC __declspec(dllimport)
#endif

#if defined(LIBXML_FASTCALL)
  #define XMLCALL __fastcall
#else
  #define XMLCALL __cdecl
#endif
#define XMLCDECL __cdecl

/** DOC_ENABLE */
#else /* not Windows */

/**
 * XMLPUBLIC:
 *
 * Macro which declares a public symbol
 */
#define XMLPUBLIC

/**
 * XMLCALL:
 *
 * Macro which declares the calling convention for exported functions
 */
#define XMLCALL

/**
 * XMLCDECL:
 *
 * Macro which declares the calling convention for exported functions that
 * use '...'.
 */
#define XMLCDECL

#endif /* platform switch */

/*
 * XMLPUBFUN:
 *
 * Macro which declares an exportable function
 */
#define XMLPUBFUN XMLPUBLIC

/**
 * XMLPUBVAR:
 *
 * Macro which declares an exportable variable
 */
#define XMLPUBVAR XMLPUBLIC extern

/* Compatibility */
#if !defined(LIBXML_DLL_IMPORT)
#define LIBXML_DLL_IMPORT XMLPUBVAR
#endif

#endif /* __XML_EXPORTS_H__ */


