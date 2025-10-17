/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 1, 2024.
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
**
**  NAME:
**
**      allocate.c
**
**  FACILITY:
**
**      IDL Stub Support Routines
**
**  ABSTRACT:
**
**  Stub memory allocation and free routines to keep track of all allocated
**  memory so that it can readily be freed
**
**  VERSION: DCE 1.0
**
*/
#if HAVE_CONFIG_H
#include <config.h>
#endif

#include <dce/rpc.h>
#include <dce/stubbase.h>
#include <lsysdep.h>

#ifdef DEBUG_VERBOSE
#   include <stdio.h>
#endif

#ifdef PERFMON
#include <dce/idl_log.h>
#endif

typedef struct memlink
{
    rpc_void_p_t obj;
    struct memlink *next;
} memlink;

rpc_void_p_t
rpc_ss_mem_alloc(rpc_ss_mem_handle *handle, size_t bytes)
{

    error_status_t status = 0;
    byte_p_t result;

    result = rpc_sm_mem_alloc(handle, bytes, &status);

    if (status == rpc_s_no_memory)
        DCETHREAD_RAISE( rpc_x_no_memory );

    return result;
}

rpc_void_p_t
rpc_sm_mem_alloc (rpc_ss_mem_handle *handle, size_t bytes, error_status_t *st)
{
    memlink* l = (memlink*) handle->alloc(sizeof(memlink));

#ifdef PERFMON
    RPC_SM_MEM_ALLOC_N;
#endif

    if (l == NULL)
    {
        *st = rpc_s_no_memory;
        return NULL;
    }

    l->obj = handle->alloc((idl_size_t) bytes);

    if (l->obj == NULL)
    {
        *st = rpc_s_no_memory;
        handle->free(l);
        return NULL;
    }

    l->next = (memlink*) handle->memory;
    handle->memory = l;

#ifdef PERFMON
    RPC_SM_MEM_ALLOC_X;
#endif

    return l->obj;
}

void
rpc_ss_mem_free (rpc_ss_mem_handle *handle)
{
    memlink* lp, *next;
#ifdef PERFMON
    RPC_SS_MEM_FREE_N;
#endif

    for (lp = (memlink*) handle->memory; lp; lp = next)
    {
        next = lp->next;
        handle->free(lp->obj);
        handle->free(lp);
    }

#ifdef PERFMON
    RPC_SS_MEM_FREE_X;
#endif

}

void
rpc_ss_mem_release (rpc_ss_mem_handle *handle, byte_p_t data_addr, int freeit)
{
    memlink** lp, **next, *memory;

#ifdef PERFMON
    RPC_SS_MEM_RELEASE_N;
#endif

    memory = (memlink*) handle->memory;
    for (lp = &memory; *lp; lp = next)
    {
        next = &(*lp)->next;

        if ((*lp)->obj == data_addr)
        {
            memlink* realnext = *next;
            if (freeit)
                handle->free((*lp)->obj);
            handle->free(*lp);
            *lp = realnext;
            break;
        }
    }
    handle->memory = (idl_void_p_t) memory;

#ifdef PERFMON
    RPC_SS_MEM_RELEASE_X;
#endif

}

#ifdef MIA
void
rpc_ss_mem_item_free (rpc_ss_mem_handle *handle, byte_p_t data_addr)
{
#ifdef PERFMON
    RPC_SS_MEM_ITEM_FREE_N;
#endif

    rpc_ss_mem_release(handle, data_addr, 1);

#ifdef PERFMON
    RPC_SS_MEM_ITEM_FREE_X;
#endif

}
#endif

#if 0
void
rpc_ss_mem_dealloc (byte_p_t data_addr)
{
#ifdef PERFMON
    RPC_SS_MEM_DEALLOC_N;
#endif

    printf("BADNESS: dealloc reached\n");

#ifdef PERFMON
    RPC_SS_MEM_DEALLOC_X;
#endif
}
#endif

#if 0
void traverse_list(rpc_ss_mem_handle handle)
{
    printf("List contains:");
    while (handle)
    {
        printf(" %d", handle);
        handle = ((header *)handle)->next;
    }
    printf(" (done)\n");
}

void main()
{
    char buf[100];
    byte_p_t tmp, *buff_addr;
    rpc_ss_mem_handle handle = NULL;

    do
    {
        printf("q/a bytes/f addr/d addr:");
        gets(buf);
        if (*buf == 'q')
        {
            rpc_ss_mem_free(&handle);
            exit();
        }
        if (*buf == 'a')
            if ((tmp = rpc_ss_mem_alloc(&handle, atoi(buf+2))) == NULL)
                printf("\tCouldn't get memory\n");
                else printf("\tGot %d\n", tmp);
        if (*buf == 'f')
            rpc_ss_mem_release(&handle, (byte_p_t)atoi(buf+2), 1);
        if (*buf == 'd')
            rpc_ss_mem_dealloc((byte_p_t)atoi(buf+2));
        traverse_list(handle);
    } while (*buf != 'q');
}
#endif
