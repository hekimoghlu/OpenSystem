/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 6, 2021.
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
#ifndef __XML_XSLTLOCALE_H__
#define __XML_XSLTLOCALE_H__

#include <libxml/xmlstring.h>
#include "xsltexports.h"

#ifdef HAVE_STRXFRM_L

/*
 * XSLT_LOCALE_POSIX:
 * Macro indicating to use POSIX locale extensions
 */
#define XSLT_LOCALE_POSIX

#ifdef HAVE_LOCALE_H
#include <locale.h>
#endif
#ifdef HAVE_XLOCALE_H
#include <xlocale.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef locale_t xsltLocale;
typedef xmlChar xsltLocaleChar;

#ifdef __cplusplus
}
#endif

#elif defined(_WIN32) && !defined(__CYGWIN__)

/*
 * XSLT_LOCALE_WINAPI:
 * Macro indicating to use WinAPI for extended locale support
 */
#define XSLT_LOCALE_WINAPI

#include <windows.h>
#include <winnls.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef LCID xsltLocale;
typedef wchar_t xsltLocaleChar;

#ifdef __cplusplus
}
#endif

#else

/*
 * XSLT_LOCALE_NONE:
 * Macro indicating that there's no extended locale support
 */
#define XSLT_LOCALE_NONE

#ifdef __cplusplus
extern "C" {
#endif

typedef void *xsltLocale;
typedef xmlChar xsltLocaleChar;

#ifdef __cplusplus
}
#endif

#endif

#ifdef __cplusplus
extern "C" {
#endif

XSLTPUBFUN xsltLocale XSLTCALL
	xsltNewLocale			(const xmlChar *langName);
XSLTPUBFUN void XSLTCALL
	xsltFreeLocale			(xsltLocale locale);
XSLTPUBFUN xsltLocaleChar * XSLTCALL
	xsltStrxfrm			(xsltLocale locale,
					 const xmlChar *string);
XSLTPUBFUN int XSLTCALL
	xsltLocaleStrcmp		(xsltLocale locale,
					 const xsltLocaleChar *str1,
					 const xsltLocaleChar *str2);
XSLTPUBFUN void XSLTCALL
	xsltFreeLocales			(void);

#ifdef __cplusplus
}
#endif

#endif /* __XML_XSLTLOCALE_H__ */
