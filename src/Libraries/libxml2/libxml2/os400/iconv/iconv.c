/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 29, 2025.
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
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>

#include "/QIBM/include/iconv.h"        /* Force system definition. */

#define USE_SYSTEM_ICONV
#include "iconv.h"                      /* Use local definitions. */



/**
***     Bring-in the name-->CCSID mapping DFA tables.
**/

#include "ianatables.c"



static int
findEncoding(const unsigned char * * namep)

{
        t_staterange curstate;
        t_ccsid ccsid;
        t_ccsid final;
        t_transrange l;
        t_transrange h;
        const unsigned char * name;

        /**
        ***     Get the CCSID correspong to the name at *`namep'.
        ***     If success, update pointer at `namep' to 1st byte after matched
        ***             name and return the CCSID.
        ***     If failure, set errno and return -1.
        **/

        if (!namep || !(name = *namep)) {
                errno = EINVAL;
                return -1;
                }

        curstate = 0;
        final = 0;

        for (;;) {
                if (curstate < sizeof final_array / sizeof final_array[0])
                        if (final_array[curstate]) {
                                final = final_array[curstate];
                                *namep = name;
                                }

                l = trans_array[curstate] - 1;
                h = trans_array[curstate + 1];

                do {
                        if (++l >= h) {
                                if (!final) {
                                        errno = EINVAL;
                                        return -1;
                                        }

                                return final - 1;
                                }
                } while (label_array[l] != *name);

                curstate = goto_array[l];
                name++;
                }

        /* NOTREACHED. */
}


static void
makeos400codename(char * buf, unsigned int ccsid)

{
        ccsid &= 0xFFFF;
        memset(buf, 0, 32);
        sprintf(buf, "IBMCCSID%05u0000000", ccsid);
}


Iconv_t
IconvOpen(const char * tocode, const char * fromcode)

{
        int toccsid = findEncoding(&tocode);
        int fromccsid = findEncoding(&fromcode);
        char fromibmccsid[33];
        char toibmccsid[33];
        iconv_t * cd;

        if (toccsid < 0 || fromccsid < 0)
                return (Iconv_t) -1;

        makeos400codename(fromibmccsid, fromccsid);
        makeos400codename(toibmccsid, toccsid);
        memset(toibmccsid + 13, 0, sizeof toibmccsid - 13);

        cd = (iconv_t *) malloc(sizeof *cd);

        if (!cd)
                return (Iconv_t) -1;

        *cd = iconv_open(toibmccsid, fromibmccsid);

        if (cd->return_value) {
                free((char *) cd);
                return (Iconv_t) -1;
                }

        return (Iconv_t) cd;
}


size_t
Iconv(Iconv_t cd, char * * inbuf, size_t * inbytesleft,
                                        char * * outbuf, size_t * outbytesleft)

{
        if (!cd || cd == (Iconv_t) -1) {
                errno = EINVAL;
                return (size_t) -1;
                }

        return iconv(*(iconv_t *) cd, inbuf, inbytesleft, outbuf, outbytesleft);
}


int
IconvClose(Iconv_t cd)

{
        if (!cd || cd == (Iconv_t) -1) {
                errno = EINVAL;
                return -1;
                }

        if (iconv_close(*(iconv_t *) cd))
                return -1;

        free((char *) cd);
        return 0;
}
