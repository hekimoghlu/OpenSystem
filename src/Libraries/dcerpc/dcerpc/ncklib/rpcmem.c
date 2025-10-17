/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 1, 2023.
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
**      rpcmem.c
**
**  FACILITY:
**
**      Remote Procedure Call (RPC)
**
**  ABSTRACT:
**
**  non-macro version of runtime memory allocator (see rpcmem.h)
**
**
*/

#include <commonp.h>

PRIVATE dce_pointer_t rpc__mem_alloc
(
    size_t size,
    unsigned32 type,
    unsigned32 flags ATTRIBUTE_UNUSED
)
{
    char *addr;

    RPC_MEM_ALLOC_IL(addr, char *, size, type, flags);

#ifdef DEBUG
    if ((type & 0xff) == rpc_g_dbg_switches[rpc_es_dbg_mem_type])
    {
        RPC_DBG_PRINTF(rpc_e_dbg_mem, 1,
	    ("(rpc__mem_alloc) type %x - %lx @ %p\n",
	    type, size, addr));
    }
    else
    {
	RPC_DBG_PRINTF(rpc_e_dbg_mem, 10,
	    ("(rpc__mem_alloc) type %x - %lx @ %p\n",
	    type, size, addr));
    }
#endif

    return addr;
}

PRIVATE dce_pointer_t rpc__mem_realloc
(
    dce_pointer_t addr,
    unsigned32 size,
    unsigned32 type,
    unsigned32 flags ATTRIBUTE_UNUSED
)
{
    RPC_MEM_REALLOC_IL(addr, dce_pointer_t, size, type, flags);

#ifdef DEBUG
    if ((type & 0xff) == rpc_g_dbg_switches[rpc_es_dbg_mem_type])
    {
        RPC_DBG_PRINTF(rpc_e_dbg_mem, 1,
	    ("(rpc__mem_realloc) type %x - %x @ %p\n",
	    type, size, addr));
    }
    else
    {
	RPC_DBG_PRINTF(rpc_e_dbg_mem, 10,
	    ("(rpc__mem_realloc) type %x - %x @ %p\n",
	    type, size, addr));
    }
#endif

    return addr;
}

PRIVATE void rpc__mem_free
(
    dce_pointer_t addr,
    unsigned32 type
)
{

#ifdef DEBUG
    if ((type & 0xff) == rpc_g_dbg_switches[rpc_es_dbg_mem_type])
    {
        RPC_DBG_PRINTF(rpc_e_dbg_mem, 1,
	    ("(rpc__mem_free) type %x @ %p\n",
	    type, addr));
    }
    else
    {
	RPC_DBG_PRINTF(rpc_e_dbg_mem, 10,
	    ("(rpc__mem_free) type %x @ %p\n",
	    type, addr));
    }
#endif

    RPC_MEM_FREE_IL(addr, type);
}
