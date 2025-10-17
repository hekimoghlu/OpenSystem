/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 5, 2023.
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
**
**  NAME
**
**      util.c
**
**  FACILITY:
**
**      Remote Procedure Call (RPC)
**
**  ABSTRACT:
**
**  Utility routines to support the performance and system exerciser
**  client and server.
**
**
*/

#include <dce/dce_error.h>
#include <perf_c.h>

#include <ctype.h>
#include <stdlib.h>


/*
 * Define the Foo and Bar type UIDS.
 */
uuid_old_t FooType =
{
    0x3c4d28ff,
    0xe000,
    0x0000,
    0x0d,
    {0x00, 0x01, 0x34, 0x22, 0x00, 0x00, 0x00}
};

uuid_old_t BarType =
{
    0x3c4d2909,
    0x1000,
    0x0000,
    0x0d,
    {0x00, 0x01, 0x34, 0x22, 0x00, 0x00, 0x00}
};

idl_uuid_t ZotType =
{
    0x007dd408,
    0x05b0,
    0x1a36,
    0xac,
    0x7e,
    {0x02, 0x60, 0x8c, 0x2f, 0xec, 0xd0}
};

/*
 * Define a couple of objects of type Foo.
 */
uuid_old_t FooObj1 =
{
    0x3c4d2911,
    0x4000,
    0x0000,
    0x0d,
    {0x00, 0x01, 0x34, 0x22, 0x00, 0x00, 0x00}
};

uuid_old_t FooObj2 =
{
    0x3c4d2da7,
    0x1000,
    0x0000,
    0x0d,
    {0x00, 0x01, 0x34, 0x22, 0x00, 0x00, 0x00}
};

/*
 * Define a couple of objects of type Bar.
 */
uuid_old_t BarObj1 =
{
    0x3c4d2dae,
    0x4000,
    0x0000,
    0x0d,
    {0x00, 0x01, 0x34, 0x22, 0x00, 0x00, 0x00}
};

uuid_old_t BarObj2 =
{
    0x3c4d2919,
    0x7000,
    0x0000,
    0x0d,
    {0x00, 0x01, 0x34, 0x22, 0x00, 0x00, 0x00}
};

/*
 * Define an object of type Zot.
 */
idl_uuid_t ZotObj =
{
    0x0053e49a,
    0x05d3,
    0x1a36,
    0x8a,
    0xd8,
    {0x02, 0x60, 0x8c, 0x2f, 0xec, 0xd0}
};

/*
 * Define an object of type "nil".  (Not to be confused with the "nil" object.)
 */
idl_uuid_t NilTypeObj =
{ /* 1c5b2910-33ab-11ca-b092-08001e01d6d5 */
    0x1c5b2910,
    0x33ab,
    0x11ca,
    0xb0,
    0x92,
    {0x08, 0x00, 0x1e, 0x01, 0xd6, 0xd5}
};

/*
 * Define the "nil" object.  (Not to be confused with an object whose type
 * simply happens to be "nil".)
 */
idl_uuid_t NilObj = {0,0,0,0,0,{0,0,0,0,0,0} };


char *authn_level_names[] =
{
    "default",
    "none",
    "connect",
    "call",
    "pkt",
    "pkt_integrity",
    "pkt_privacy",
    NULL
};

char *authn_names[] =
{
    "none",
    "dce_private",
    "dce_public",
    "dce_dummy",
    NULL
};

char *authz_names[] =
{
    "none",
    "name",
    "dce",
    NULL
};

/*
 * Return error text.
 */

char *error_text (st)

unsigned32      st;

{
    static dce_error_string_t error_string;
    int inq_st;

    dce_error_inq_text (st, error_string, &inq_st);
    return ((char *) error_string);
}

/*
 * Dump storage usage info.
 */
void dump_stg_info(void)

{
#ifdef ETEXT_EDATA
    extern char etext, edata, end;

    printf ("\netext = %08x, edata = %08x, end = %08x\n\n",
        &etext, &edata, &end);
#endif
}


/*
 * Lookup a name in a table.
 */
extern void usage(int);
int lookup_name(char *table[], char *s)
{
    int i;

    if (isdigit(s[0]))
    {
        return (atoi(s));
    }

    for (i = 0; table[i] != NULL; i++)
    {
        if (strcmp(table[i], s) == 0)
        {
            return (i);
        }
    }

    usage(-1);
	 /* NOTREACHED */
	 return -1;
}
