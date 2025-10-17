/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 28, 2025.
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
*   Copyright (C) 1998-2008, International Business Machines
*   Corporation and others.  All Rights Reserved.
*
*******************************************************************************
*
* File util.c
*
* Modification History:
*
*   Date        Name        Description
*   06/10/99    stephen     Creation.
*   02/07/08    Spieth      Correct XLIFF generation on EBCDIC platform
*   
*******************************************************************************
*/

#include "unicode/putil.h"
#include "rbutil.h"
#include "cmemory.h"
#include "cstring.h"


/* go from "/usr/local/include/curses.h" to "/usr/local/include" */
void
get_dirname(char *dirname,
            const char *filename)
{
  const char *lastSlash = uprv_strrchr(filename, U_FILE_SEP_CHAR);
  if (lastSlash != NULL) {
    lastSlash++;
  }

  if(lastSlash>filename) {
    uprv_strncpy(dirname, filename, (lastSlash - filename));
    *(dirname + (lastSlash - filename)) = '\0';
  } else {
    *dirname = '\0';
  }
}

/* go from "/usr/local/include/curses.h" to "curses" */
void
get_basename(char *basename,
             const char *filename)
{
  /* strip off any leading directory portions */
  const char *lastSlash = uprv_strrchr(filename, U_FILE_SEP_CHAR);
  if (lastSlash != NULL) {
    lastSlash++;
  }
  char *lastDot;

  if(lastSlash>filename) {
    uprv_strcpy(basename, lastSlash);
  } else {
    uprv_strcpy(basename, filename);
  }

  /* strip off any suffix */
  lastDot = uprv_strrchr(basename, '.');

  if(lastDot != NULL) {
    *lastDot = '\0';
  }
}

#define MAX_DIGITS 10
int32_t 
itostr(char * buffer, int32_t i, uint32_t radix, int32_t pad)
{
    const char digits[16] = {'0','1','2','3','4','5','6','7','8','9','a','b','c','d','e','f'};
    int32_t length = 0;
    int32_t num = 0;
    int32_t save = i;
    int digit;
    int32_t j;
    char temp;
    
    /* if i is negative make it positive */
    if(i<0){
        i=-i;
    }
    
    do{
        digit = (int)(i % radix);
        buffer[length++]= digits[digit];
        i=i/radix;
    } while(i);

    while (length < pad){
        buffer[length++] = '0';/*zero padding */
    }
    
    /* if i is negative add the negative sign */
    if(save < 0){
        buffer[length++]='-';
    }

    /* null terminate the buffer */
    if(length<MAX_DIGITS){
        buffer[length] =  0x0000;
    }

    num= (pad>=length) ? pad :length;
 

    /* Reverses the string */
    for (j = 0; j < (num / 2); j++){
        temp = buffer[(length-1) - j];
        buffer[(length-1) - j] = buffer[j];
        buffer[j] = temp;
    }
    return length;
}
