/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 4, 2022.
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

// Â© 2016 and later: Unicode, Inc. and others.
// License & terms of use: http://www.unicode.org/copyright.html
/*
*******************************************************************************
*
*   Copyright (C) 1998-2011, International Business Machines
*   Corporation and others.  All Rights Reserved.
*
*******************************************************************************
*
* File read.h
*
* Modification History:
*
*   Date        Name        Description
*   05/26/99    stephen     Creation.
*   5/10/01     Ram         removed ustdio dependency
*******************************************************************************
*/

#ifndef READ_H
#define READ_H 1

#include "unicode/utypes.h"
#include "ustr.h"
#include "ucbuf.h"

/* The types of tokens which may be returned by getNextToken.
   NOTE: Keep these in sync with tokenNames in parse.c */
enum ETokenType
{
    TOK_STRING,          /* A string token, such as "MonthNames" */
    TOK_OPEN_BRACE,      /* An opening brace character */
    TOK_CLOSE_BRACE,     /* A closing brace character */
    TOK_COMMA,           /* A comma */
    TOK_COLON,           /* A colon */

    TOK_EOF,             /* End of the file has been reached successfully */
    TOK_ERROR,           /* An error, such an unterminated quoted string */
    TOK_TOKEN_COUNT      /* Number of "real" token types */
};

U_CFUNC UChar32 unescape(UCHARBUF *buf, UErrorCode *status);

U_CFUNC void resetLineNumber(void);

U_CFUNC enum ETokenType
getNextToken(UCHARBUF *buf,
             struct UString *token,
             uint32_t *linenumber, /* out: linenumber of token */
             struct UString *comment,
             UErrorCode *status);

#endif
