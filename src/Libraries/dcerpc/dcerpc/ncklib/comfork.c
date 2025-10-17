/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 23, 2024.
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
**      rpc/runtime/comfork.c
**
**  FACILITY:
**
**      Remote Procedure Call (RPC)
**
**  ABSTRACT:
**
**  Include pthreads atfork grodiness.
**
**
*/

#include <commonp.h>
#include <com.h>        /* Externals for Common Services component  */

#if defined(ATFORK_SUPPORTED)

PRIVATE boolean32 rpc__fork_is_in_progress (void);

/*
 * Boolean indicating whether fork is in progress or not.
 */
INTERNAL volatile boolean32 atfork_in_progress = false;

/*
 * This really doesn't belong here, but is here for pragmatic reasons.
 */

typedef void (*atfork_handler_ptr_t) (rpc_fork_stage_id_t /*stage*/);

/*
 * Save the address of the fork handler registered through atfork.
 * This pointer should be saved ONLY once by rpc__atfork(). If
 * rpc__atfork() is called more than once, which happens in the
 * child context when rpc__init() is called, do not register the
 * fork handler.
 */

/*
 * Private handler state for the RPC runtime
 */
INTERNAL volatile atfork_handler_ptr_t  atfork_handler_ptr = NULL;

INTERNAL struct atfork_user_data_t
{
    pid_t       pid;    /* who registered */
    unsigned32  pre;    /* How many times _pre_fork called */
    unsigned32  post_p; /*                _post_fork_parent called */
    unsigned32  post_c; /*                _post_fork_child called */
} rpc__atfork_user_data;

/* ========================================================================= */

INTERNAL void _pre_fork (
        dce_pointer_t   /*arg*/
    );

INTERNAL void _post_fork_child (
        dce_pointer_t   /*arg*/
    );

INTERNAL void _post_fork_parent (
        dce_pointer_t   /*arg*/
    );

/* ========================================================================= */

/*
 * _ P R E _ F O R K
 *
 * This procedure is called by the Pthreads library prior to calling
 * the fork/vfork system call.
 */
INTERNAL void _pre_fork
(
  dce_pointer_t arg
)
{
    RPC_DBG_PRINTF(rpc_e_dbg_atfork, 1,
               ("(_pre_fork) entering, pid %d, pre %d, post_p %d, post_c %d\n",
                    ((struct atfork_user_data_t *) arg)->pid,
                    ((struct atfork_user_data_t *) arg)->pre,
                    ((struct atfork_user_data_t *) arg)->post_p,
                    ((struct atfork_user_data_t *) arg)->post_c));

    atfork_in_progress = true;
    ((struct atfork_user_data_t *) arg)->pre++;

    if (rpc_g_initialized == true)
        (*atfork_handler_ptr)(RPC_C_PREFORK);

    RPC_DBG_PRINTF(rpc_e_dbg_atfork, 1,
              ("(_pre_fork) returning, pid %d, pre %d, post_p %d, post_c %d\n",
                    ((struct atfork_user_data_t *) arg)->pid,
                    ((struct atfork_user_data_t *) arg)->pre,
                    ((struct atfork_user_data_t *) arg)->post_p,
                    ((struct atfork_user_data_t *) arg)->post_c));
}

/*
 * _ P O S T _ F O R K _ C H I L D
 *
 * This procedure is called in the child of a forked process immediately
 * after the fork is performed.
 */

INTERNAL void _post_fork_child
(
  dce_pointer_t arg
)
{
    RPC_DBG_PRINTF(rpc_e_dbg_atfork, 1,
        ("(_post_fork_child) entering, pid %d, pre %d, post_p %d, post_c %d\n",
                    ((struct atfork_user_data_t *) arg)->pid,
                    ((struct atfork_user_data_t *) arg)->pre,
                    ((struct atfork_user_data_t *) arg)->post_p,
                    ((struct atfork_user_data_t *) arg)->post_c));

    ((struct atfork_user_data_t *) arg)->post_c++;

    if (rpc_g_initialized == true)
        (*atfork_handler_ptr)(RPC_C_POSTFORK_CHILD);

    atfork_in_progress = false;

    RPC_DBG_PRINTF(rpc_e_dbg_atfork, 1,
       ("(_post_fork_child) returning, pid %d, pre %d, post_p %d, post_c %d\n",
                    ((struct atfork_user_data_t *) arg)->pid,
                    ((struct atfork_user_data_t *) arg)->pre,
                    ((struct atfork_user_data_t *) arg)->post_p,
                    ((struct atfork_user_data_t *) arg)->post_c));
}

/*
 * _ P O S T _ F O R K _ P A R E N T
 *
 * This procedure is called in the parent of a forked process immediately
 * after the fork is performed.
 */

INTERNAL void _post_fork_parent
(
  dce_pointer_t arg
)
{
    RPC_DBG_PRINTF(rpc_e_dbg_atfork, 1,
       ("(_post_fork_parent) entering, pid %d, pre %d, post_p %d, post_c %d\n",
                    ((struct atfork_user_data_t *) arg)->pid,
                    ((struct atfork_user_data_t *) arg)->pre,
                    ((struct atfork_user_data_t *) arg)->post_p,
                    ((struct atfork_user_data_t *) arg)->post_c));

    ((struct atfork_user_data_t *) arg)->post_p++;

    if (rpc_g_initialized == true)
        (*atfork_handler_ptr)(RPC_C_POSTFORK_PARENT);

    atfork_in_progress = false;

    RPC_DBG_PRINTF(rpc_e_dbg_atfork, 1,
      ("(_post_fork_parent) returning, pid %d, pre %d, post_p %d, post_c %d\n",
                    ((struct atfork_user_data_t *) arg)->pid,
                    ((struct atfork_user_data_t *) arg)->pre,
                    ((struct atfork_user_data_t *) arg)->post_p,
                    ((struct atfork_user_data_t *) arg)->post_c));
}

PRIVATE void rpc__atfork
(
 void *handler
)
{
    if (handler == NULL)
        return;

    /*
     * Don't register it again!
     */
    if (atfork_handler_ptr != NULL)
        return;

    /*
     * Save the address of the handler routine, and register our own
     * handlers. (see note above)
     */
    atfork_handler_ptr = (atfork_handler_ptr_t) handler;
    rpc__atfork_user_data.pid = getpid();
    rpc__atfork_user_data.pre = 0;
    rpc__atfork_user_data.post_p = 0;
    rpc__atfork_user_data.post_c = 0;

    dcethread_atfork_throw((dce_pointer_t)(&rpc__atfork_user_data), _pre_fork,
            _post_fork_parent, _post_fork_child);
}


/*
**++
**
**  ROUTINE NAME:       rpc__fork_is_in_progress
**
**  SCOPE:              PRIVATE - declared in com.h
**
**  DESCRIPTION:
**
**  Return true iff the atfork handler is in progress.
**
**
**  INPUTS:             none
**
**  INPUTS/OUTPUTS:     none
**
**  OUTPUTS:            none
**
**  IMPLICIT INPUTS:    none
**
**  IMPLICIT OUTPUTS:   none
**
**  FUNCTION VALUE:     boolean
**
**  SIDE EFFECTS:       none
**
**--
**/

PRIVATE boolean32 rpc__fork_is_in_progress (void)
{
    /*
     * We could lock, but there should be only one thread in fork-progress
     * which can change it.
     */
    return (atfork_in_progress);
}

#endif  /* defined(ATFORK_SUPPORTED) */
