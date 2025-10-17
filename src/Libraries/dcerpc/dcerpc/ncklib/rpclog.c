/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 31, 2025.
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
**      rpclog.c
**
**  FACILITY:
**
**      Remote Procedure Call (RPC)
**
**  ABSTRACT:
**
**  Definitions of global variables.
**
**
*/

#if HAVE_CONFIG_H
#include <config.h>
#endif

#ifdef LOGPTS
#include <rpclog.h>

#ifdef ultrix
#include <nlist.h>
#include <unistd.h>
#endif /* ultrix */

static rpc_logpt_t      logpt_invisible;
rpc_logpt_ptr_t         rpc_g_log_ptr = &logpt_invisible;

/*
**++
**
**  ROUTINE NAME:       rpc__log_ptr_init
**
**  SCOPE:              PRIVATE - declared in rpclog.h
**
**  DESCRIPTION:
**
**  This routine will initialize the RPC logging service.
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
**  FUNCTION VALUE:
**
**      log pointer     pointer value to location to which codes to
**                      be timestamped are written.
**
**  SIDE EFFECTS:       none
**
**--
**/

rpc_logpt_ptr_t rpc__log_ptr_init (void)
#ifdef ultrix
{
    rpc_logpt_ptr_t     ptr;
    unsigned long       logpt_addr_in_virt_mem;

#define QMEM_X 0
    struct nlist symtab[QMEM_X + 2];

    symtab[QMEM_X].n_name = "_qmem";
    symtab[QMEM_X + 1].n_name = NULL;

    nlist ("/vmunix", symtab);
    logpt_addr_in_virt_mem = (symtab[QMEM_X].n_value + LOGPT_ADDR_IN_QMEM);
    ptr = (rpc_logpt_ptr_t) (logpt_addr_in_virt_mem);

    return (ptr);
}

#endif /* ultrix */

#else
#ifndef __GNUC__
/*
 *  ANSI c does not allow a file to be compiled without declarations.
 *  If LOGPTS is not defined, we need to declare a dummy variable to
 *  compile under strict ansi c standards.
 */
static  char    _rpclog_dummy_ = 0, *_rpclog_dummy_p = &_rpclog_dummy_;
#endif
#endif /* LOGOPTS */
