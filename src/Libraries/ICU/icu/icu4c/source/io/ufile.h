/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 19, 2024.
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
 *   Copyright (C) 1998-2014, International Business Machines
 *   Corporation and others.  All Rights Reserved.
 *
 *******************************************************************************
 *
 * File ufile.h
 *
 * Modification History:
 *
 *   Date        Name        Description
 *   12/01/98    stephen        Creation.
 *   03/12/99    stephen     Modified for new C API.
 *******************************************************************************
 */

#ifndef UFILE_H
#define UFILE_H

#include "unicode/utypes.h"

#if !UCONFIG_NO_CONVERSION

#include <stdio.h>

#include "unicode/ucnv.h"
#include "unicode/utrans.h"
#include "locbund.h"

/* The buffer size for fromUnicode calls */
#define UFILE_CHARBUFFER_SIZE 1024

/* The buffer size for toUnicode calls */
#define UFILE_UCHARBUFFER_SIZE 1024

/* A UFILE */

#if !UCONFIG_NO_TRANSLITERATION

typedef struct {
    UChar  *buffer;             /* Beginning of buffer */
    int32_t capacity;           /* Capacity of buffer */
    int32_t pos;                /* Beginning of untranslitted data */
    int32_t length;             /* Length *from beginning of buffer* of untranslitted data */
    UTransliterator *translit;
} UFILETranslitBuffer;

#endif

typedef struct u_localized_string {
    UChar       *fPos;          /* current pos in fUCBuffer */
    const UChar *fLimit;        /* data limit in fUCBuffer */
    UChar       *fBuffer;       /* Place to write the string */

#if !UCONFIG_NO_FORMATTING
    ULocaleBundle  fBundle; /* formatters */
#endif
} u_localized_string;

struct UFILE {
#if !UCONFIG_NO_TRANSLITERATION
    UFILETranslitBuffer *fTranslit;
#endif

    FILE        *fFile;         /* the actual filesystem interface */

    UConverter  *fConverter;    /* for codeset conversion */

    u_localized_string str;     /* struct to handle strings for number formatting */

    UChar       fUCBuffer[UFILE_UCHARBUFFER_SIZE];/* buffer used for toUnicode */

    UBool       fOwnFile;       /* true if fFile should be closed */

    int32_t     fFileno;        /* File number. Useful to determine if it's stdin. */
};

/**
 * Like u_file_write but takes a flush parameter
 */
U_CFUNC int32_t U_EXPORT2
u_file_write_flush( const UChar     *chars, 
        int32_t     count, 
        UFILE       *f,
        UBool       flushIO,
        UBool       flushTranslit);

/**
 * Fill a UFILE's buffer with converted codepage data.
 * @param f The UFILE containing the buffer to fill.
 */
void
ufile_fill_uchar_buffer(UFILE *f);

/**
 * Get one code unit and detect whether the end of file has been reached.
 * @param f The UFILE containing the characters.
 * @param ch The read in character
 * @return true if the character is valid, or false when EOF has been detected
 */
U_CFUNC UBool U_EXPORT2
ufile_getch(UFILE *f, UChar *ch);

/**
 * Get one character and detect whether the end of file has been reached.
 * @param f The UFILE containing the characters.
 * @param ch The read in character
 * @return true if the character is valid, or false when EOF has been detected
 */
U_CFUNC UBool U_EXPORT2
ufile_getch32(UFILE *f, UChar32 *ch);

/**
 * Close out the transliterator and flush any data therein.
 * @param f flu
 */
void 
ufile_close_translit(UFILE *f);

/**
 * Flush the buffer in the transliterator 
 * @param f UFile to flush
 */
void 
ufile_flush_translit(UFILE *f);

/**
 * Flush the IO buffer
 * @param f UFile to flush
 */
void 
ufile_flush_io(UFILE *f);


#endif
#endif
