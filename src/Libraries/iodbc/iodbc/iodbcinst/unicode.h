/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 30, 2024.
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
#ifndef _UNICODE_H
#define _UNICODE_H

#if HAVE_WCHAR_H
#include <wchar.h>
#endif


/*
 *  Max length of a UTF-8 encoded character sequence
 */
#define UTF8_MAX_CHAR_LEN 4

/*
 *  Function Prototypes
 */
SQLCHAR *dm_SQL_W2A (SQLWCHAR * inStr, ssize_t size);
SQLCHAR *dm_SQL_WtoU8 (SQLWCHAR * inStr, ssize_t size);
SQLCHAR *dm_strcpy_W2A (SQLCHAR * destStr, SQLWCHAR * sourStr);

SQLWCHAR *dm_SQL_A2W (SQLCHAR * inStr, ssize_t size);
SQLWCHAR *dm_SQL_U8toW (SQLCHAR * inStr, SQLSMALLINT size);
SQLWCHAR *dm_strcpy_A2W (SQLWCHAR * destStr, SQLCHAR * sourStr);

int dm_StrCopyOut2_A2W (SQLCHAR * inStr, SQLWCHAR * outStr, SQLSMALLINT size,
    SQLSMALLINT * result);
int dm_StrCopyOut2_U8toW (SQLCHAR * inStr, SQLWCHAR * outStr, size_t size,
    u_short * result);
int dm_StrCopyOut2_W2A (SQLWCHAR * inStr, SQLCHAR * outStr, SQLSMALLINT size,
    SQLSMALLINT * result);


# ifdef WIN32
#define OPL_W2A(w, a, cb)     \
	WideCharToMultiByte(CP_ACP, 0, w, cb, a, cb, NULL, NULL)

#define OPL_A2W(a, w, cb)     \
	MultiByteToWideChar(CP_ACP, 0, a, cb, w, cb)

# else
#define OPL_W2A(XW, XA, SIZE)      wcstombs((char *) XA, (wchar_t *) XW, SIZE)
#define OPL_A2W(XA, XW, SIZE)      mbstowcs((wchar_t *) XW, (char *) XA, SIZE)
# endif

#if MACOSX >= 103
#define HAVE_WCHAR_H 
#define HAVE_WCSLEN 
#define HAVE_WCSCPY 
#define HAVE_WCSNCPY
#define HAVE_WCSCHR
#define HAVE_WCSCAT
#define HAVE_WCSCMP
#define HAVE_TOWLOWER
#endif

/*
 *  Replacement functions
 */
#if !defined(HAVE_WCSLEN)
size_t wcslen (const wchar_t * wcs);
#endif
#if !defined(HAVE_WCSCPY)
wchar_t * wcscpy (wchar_t * wcd, const wchar_t * wcs);
#endif
#if !defined(HAVE_WCSNCPY)
wchar_t * wcsncpy (wchar_t * wcd, const wchar_t * wcs, size_t n);
#endif
#if !defined(HAVE_WCSCHR)
wchar_t* wcschr(const wchar_t *wcs, const wchar_t wc);
#endif
#if !defined(HAVE_WCSCAT)
wchar_t* wcscat(wchar_t *dest, const wchar_t *src);
#endif
#if !defined(HAVE_WCSCMP)
int wcscmp (const wchar_t* s1, const wchar_t* s2);
#endif
#if !defined(HAVE_WCSNCASECMP)
int wcsncasecmp (wchar_t* s1, wchar_t* s2, size_t n);
#endif

#endif /* _UNICODE_H */
