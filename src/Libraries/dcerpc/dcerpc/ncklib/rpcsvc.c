/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 11, 2023.
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
#include "config.h"
#include <ctype.h>
#include <commonp.h>
#include <string.h>
#include <rpcsvc.h>
#include <stdarg.h>

#if HAVE_CRASHREPORTERCLIENT_H

#include <CrashReporterClient.h>

#elif defined(__APPLE__)

/*
 * The following symbol is reference by Crash Reporter symbolicly
 * (instead of through undefined references. To get strip(1) to know
 * this symbol is not to be stripped it needs to have the
 * REFERENCED_DYNAMICALLY bit (0x10) set.  This would have been done
 * automaticly by ld(1) if this symbol were referenced through undefined
 * symbols.
 *
 * NOTE: this is an unsupported interface and the CrashReporter team reserve
 * the right to change it at any time.
 */
char *__crashreporter_info__ = NULL;
asm(".desc ___crashreporter_info__, 0x10");

#define CRSetCrashLogMessage(msg) do { \
    __crashreporter_info__ = (msg); \
} while (0)

#else

/* No CrashReporter support, spit it out to stderr and hope someone is
 * watching.
 */
#define CRSetCrashLogMessage(msg) do { \
    write(STDERR_FILENO, strlen(msg), msg); \
} while (0)

#endif

/*
dce_svc_handle_t rpc_g_svc_handle;
*/
//DCE_SVC_DEFINE_HANDLE(rpc_g_svc_handle, rpc_g_svc_table, "rpc")

//#define RPC_DCE_SVC_PRINTF(args) rpc_dce_svc_printf(args)

void rpc_dce_svc_printf (
                        const char* file,
                        unsigned int line,
                        const char *format,
                        unsigned32 dbg_switch ATTRIBUTE_UNUSED,
                        unsigned32 sev_action_flags,
                        unsigned32 error_code,
                        ... )
{
    char buff[1024];
    char *s = buff;
    size_t remain = sizeof(buff);
    va_list arg_ptr;
    int cs;

    snprintf (s, remain, "[file %s, line %d] ", file, line);
    s = &buff[strlen(buff)];
    remain = sizeof(buff) - (s - buff);

    snprintf (s, remain, "[flags: 0x%x] ", (unsigned int) sev_action_flags);
    s = &buff[strlen(buff)];
    remain = sizeof(buff) - (s - buff);

    snprintf (s, remain, "[error: 0x%x] ", (unsigned int) error_code);
    s = &buff[strlen(buff)];
    remain = sizeof(buff) - (s - buff);

    va_start (arg_ptr, error_code);
    vsnprintf (s, remain, format, arg_ptr);
    va_end (arg_ptr);

    if ( (sev_action_flags & svc_c_action_abort) ||
        (sev_action_flags & svc_c_action_exit_bad) )
    {
        CRSetCrashLogMessage(buff);
        abort();
    }
    else
    {
        cs = dcethread_enableinterrupt_throw(0);
        dcethread_write (2, buff, strlen (buff));
        dcethread_enableinterrupt_throw(cs);
    }
}

#if 0
/*
 * R P C _ _ S V C _ E P R I N T F
 *
 * Format and print arguments as a serviceability
 * debug message.
 */

PRIVATE int rpc__svc_eprintf ( char *fmt, ... )
{
    char	buf[RPC__SVC_DBG_MSG_SZ];
    va_list	arg_ptr;

    va_start (arg_ptr, fmt);
    vsprintf (buf, fmt, arg_ptr);
    va_end (arg_ptr);
    DCE_SVC_DEBUG((RPC__SVC_HANDLE, rpc_svc_general, RPC__SVC_DBG_LEVEL(0), buf));
    return(0);
}


/*
 * R P C _ _ S V C _ I N I T
 *
 * Do initialization required for serviceability
 */

PRIVATE void rpc__svc_init ( void )
{
    error_status_t status;

    /*
     * Currently, all we have to do is return, since
     * everything is statically registered.
     *
     * But someday we might do something like turn
     * on debug levels corresponding to things set
     * in rpc_g_dbg_switches[], or ...
     */

    /*
     * This silliness is a placeholder, so that we
     * remember to do things differently in the kernel
     * if we ever decide to do more than just return
     * out of this routine.
     */
    return;
}

/*
 * R P C _ _ S V C _ F M T _ D B G _ M S G
 *
 * This routine takes the printf "pargs" passed to
 * the RPC_DBG_PRINTF() macro and formats them
 * into a string that can be handed to DCE_SVC_DEBUG.
 *
 * This is necessary because the pargs are passed
 * in as a single, parenthesized argument -- which
 * also requires that the resulting string be passed
 * back as a pointer return value.
 *
 * The returned pointer must be free()'ed by the
 * caller (see comments at malloc() below).  This
 * should be fairly safe, since this routine should
 * only ever be called by RPC_DBG_PRINTF.
 */

PRIVATE char * rpc__svc_fmt_dbg_msg (char *format, ...)
{
    char            *bptr;
    va_list         arg_ptr;

    /*
     * Using malloc here is ugly but necessary.  The formatted
     * string must be passed back as a pointer return value.  The
     * possibility of recursive calls due to evaluation of pargs
     * (where, e.g., one of the pargs is a call to a routine that
     * calls RPC_DBG_PRINTF) preclude an implementation using a
     * mutex to protect a static buffer.  The potential for infinite
     * recursion precludes allocating memory using internal RPC
     * interfaces, since those interfaces call RPC_DBG_PRINTF.
     */

    if( (bptr = malloc(RPC__SVC_DBG_MSG_SZ*sizeof(char))) == NULL )
    {
        /* die horribly */
        abort();
    }

    va_start (arg_ptr, format);
    vsprintf (bptr, format, arg_ptr);
    va_end (arg_ptr);

    return( bptr );
}
#endif	/* DEBUG */
