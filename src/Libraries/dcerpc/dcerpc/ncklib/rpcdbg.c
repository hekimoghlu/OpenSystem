/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 25, 2023.
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
**  NAME:
**
**      rpcdbg.c
**
**  FACILITY:
**
**      Remote Procedure Call (RPC)
**
**  ABSTRACT:
**
**  Various data and functions for the debug component.
**
**
*/
#if HAVE_CONFIG_H
#include <config.h>
#endif

#include <ctype.h>
#include <commonp.h>
#include <string.h>

/*
 * Debug table
 *
 * A vector of "debug levels", one level per "debug switch".
 */

GLOBAL unsigned8 rpc_g_dbg_switches[RPC_DBG_N_SWITCHES];

/*
 * string buffer used by uuid_string()
 */
#ifdef DEBUG
INTERNAL char         uuid_string_buff[40];
#endif

/*
 * Make more allowances for (kernel) portability.
 */
#ifndef RPC_DBG_PRINTF_STDERR
#  define RPC_DBG_PRINTF_STDERR     fprintf(stderr,
#endif

/*
 * R P C _ _ D B G _ S E T _ S W I T C H E S
 *
 * Set debug switches from string.  The format of the string is:
 *
 *      SwitchRange.level,SwitchRange.level,...
 *
 * where a "SwitchRange" is either a single integer (e.g., "5") or a
 * range of integers of the form "integer-integer" (e.g., "1-5").  "level"
 * is the integer debug level to be applied to all the switches in the
 * range.  Putting it all together, an sample string that this function
 * can process is: "1-5.7,6.3,9.5". *
 *
 * This code largely cribbed from sendmail's "tTflag" function, which is...
 *
 * Copyright (c) 1988 Regents of the University of California.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms are permitted
 * provided that this notice is preserved and that due credit is given
 * to the University of California at Berkeley. The name of the University
 * may not be used to endorse or promote products derived from this
 * software without specific prior written permission. This software
 * is provided ``as is'' without express or implied warranty.
 *
 *  Sendmail
 *  Copyright (c) 1983  Eric P. Allman
 *  Berkeley, California
 */

PUBLIC void rpc__dbg_set_switches
(
    const char      *s ATTRIBUTE_UNUSED,
    unsigned32      *status
)
{
    int         first, last;
    register int i;

    *status = rpc_s_ok;

    for (;;)
    {
        /*
         * find first flag to set
         */
        i = 0;
        while (isdigit ((int) *s))
            i = i * 10 + (*s++ - '0');
        first = i;

        /*
         * find last flag to set
         */
        if (*s == '-')
        {
            i = 0;
            while (isdigit ((int) *++s))
                i = i * 10 + (*s - '0');
        }
        last = i;

        /*
         * find the level to set it to
         */
        i = 1;
        if (*s == '.')
        {
            i = 0;
            while (isdigit ((int) *++s))
                i = i * 10 + (*s - '0');
        }

        /*
         * clean up args
         */
        if (first >= RPC_DBG_N_SWITCHES)
            first = RPC_DBG_N_SWITCHES - 1;
        if (last >= RPC_DBG_N_SWITCHES)
            last = RPC_DBG_N_SWITCHES - 1;

        /*
         * set the flags
         */
        while (first <= last)
            rpc_g_dbg_switches[first++] = i;

        /*
         * more arguments?
         */
        if (*s++ == '\0')
            return;
    }
}


/* ======================================================================= */

#ifndef DCE_RPC_SVC
#ifndef NO_RPC_PRINTF

/*
 * R P C _ _ P R I N T F
 *
 * Note: This function uses a variable-length argument list. The "right"
 * way to handle this is using the ANSI C notation (listed below under
 * #ifdef STDARG_PRINTF). However, not all of the compilers support this,
 * so it's here just for future reference purposes.
 *
 * An alternative is to use the "varargs" convention (listed below under
 * #ifndef NO_VARARGS_PRINTF). Most compilers support this convention,
 * however you can't use prototypes with this.
 *
 * The last choice is to use the "old" notation. In this case also you
 * can't use prototypes.
 *
 * Only support the stdargs form
 */
PRIVATE ssize_t rpc__printf (const char *format, ...)
{
    char buff[1024];
    char *s = buff;
    size_t remain = sizeof(buff);

    if (RPC_DBG (rpc_e_dbg_pid, 1))
    {
        snprintf (s, remain, "[pid: %06lu] ", (unsigned long)getpid());
        s = &buff[strlen(buff)];
        remain = sizeof(buff) - (s - buff);
    }

    if (RPC_DBG (rpc_e_dbg_timestamp, 1))
    {
        snprintf (s, remain, "[time: %06lu] ", (unsigned long) rpc__clock_stamp());
        s = &buff[strlen(buff)];
        remain = sizeof(buff) - (s - buff);
    }

    if (RPC_DBG (rpc_e_dbg_thread_id, 1))
    {
        dcethread* self;

        self = dcethread_self ();
#ifdef CMA_INCLUDE
        snprintf (s, remain, "[thread: %08x.%08x] ", self.field1, self.field2);
#else
        snprintf (s, remain, "[thread: %08lx] ", (unsigned long) self);
#endif
        s = &buff[strlen (buff)];
        remain = sizeof(buff) - (s - buff);
    }

    {
	va_list         arg_ptr;

	va_start (arg_ptr, format);
	vsnprintf (s, remain, format, arg_ptr);
	va_end (arg_ptr);
    }

    {
        int cs;
        ssize_t ret;

        cs = dcethread_enableinterrupt_throw(0);
        ret = dcethread_write (2, buff, strlen (buff));
        dcethread_enableinterrupt_throw(cs);
        if (ret < 0)
            return ret;
    }
    return 0;
}

#endif /* NO_RPC_PRINTF */
#endif /* DCE_RPC_SVC */

/*
 * R P C _ _ D I E
 *
 * Try to report what happened and get out.
 *
 */

PRIVATE void rpc__die
(
    char            *text,
    char            *file,
    int             line
)
{
#ifndef FILE_SEPARATOR_CHAR
#define FILE_SEPARATOR_CHAR '/'
/*#error  "FILE_SEPARATOR_CHAR not defined!"*/
#endif

    char        *p = strrchr (file, FILE_SEPARATOR_CHAR);

    EPRINTF("(rpc) *** FATAL ERROR \"%s\" at %s\\%d ***\n",
            text, p == NULL ? file : p + 1, line);
    abort ();
}


/*
 * R P C _ _ U U I D _ S T R I N G
 *
 * Return a pointer to a printed UUID.
 */

PRIVATE char *rpc__uuid_string
(
    idl_uuid_t          *uuid ATTRIBUTE_UNUSED
)
{
#ifndef DEBUG

    return ("");

#else

    unsigned_char_p_t   uuid_string_p;
    unsigned32          status;

    uuid_to_string (uuid, &uuid_string_p, &status);
    if (status != uuid_s_ok)
    {
        return (NULL);
    }

    strncpy (uuid_string_buff, (char *) uuid_string_p, sizeof uuid_string_buff);
    rpc_string_free (&uuid_string_p, &status);

    return (uuid_string_buff);

#endif
}


#if !defined(DCE_RPC_SVC)
/*
 * R P C _ _ P R I N T _ S O U R C E
 *
 * Auxiliary function to print source file name and line number.  Used by
 * RPC_DBG_PRINT macro.
 */

PRIVATE void rpc__print_source
(
    const char      *file ATTRIBUTE_UNUSED,
    int             line ATTRIBUTE_UNUSED
)
{
    if (RPC_DBG(rpc_e_dbg_source, 1))
    {
        EPRINTF("    [file: %s, line: %d]\n", file, line);
    }
}
#endif /* !DCE_RPC_SVC */
