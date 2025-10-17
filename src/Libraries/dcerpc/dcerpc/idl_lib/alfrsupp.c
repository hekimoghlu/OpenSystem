/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 28, 2024.
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
**      alfrsupp.c
**
**  FACILITY:
**
**      IDL Stub Runtime Support
**
**  ABSTRACT:
**
**      Support routines for rpc_ss_allocate, rpc_ss_free
**
**  VERSION: DCE 1.0
*/

#if HAVE_CONFIG_H
#include <config.h>
#endif

#include <dce/rpc.h>
#include <dce/stubbase.h>
#ifdef MIA
#include <dce/idlddefs.h>
#endif
#include <lsysdep.h>
#include <assert.h>
/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 5, 2023.
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
 */#ifdef STUBS_USE_PTHREADS
typedef void (*destructor_t)
(
    rpc_ss_threads_dest_arg_t arg
);
#else
    typedef cma_t_destructor destructor_t;
#endif

/******************************************************************************/
/*                                                                            */
/*    Set up CMA machinery required by rpc_ss_allocate, rpc_ss_free           */
/*                                                                            */
/******************************************************************************/
ndr_boolean rpc_ss_allocate_is_set_up = ndr_false;

static RPC_SS_THREADS_ONCE_T allocate_once = RPC_SS_THREADS_ONCE_INIT;

RPC_SS_THREADS_KEY_T rpc_ss_thread_supp_key;

/******************************************************************************/
/*                                                                            */
/*    rpc_ss_destroy_thread_ctx                                               */
/*    Destroy client per thread context.                                      */
/*                                                                            */
/******************************************************************************/
static void rpc_ss_destroy_thread_ctx
(
    rpc_ss_thread_indirection_t *thread_indirection_ptr
)
{
    rpc_ss_thread_support_ptrs_t *p_thread_support_ptrs;

    if (thread_indirection_ptr != NULL)
    {
        if (thread_indirection_ptr->free_referents)
        {
            p_thread_support_ptrs = thread_indirection_ptr->indirection;

            /* Release any memory owned by the memory handle */
            rpc_ss_mem_free( p_thread_support_ptrs->p_mem_h );

            /*
             *  Free the objects it points at.
             *  Must cast because instance_of
             *  (rpc_ss_thread_support_ptrs_t).p_mem_h
             *  is of type rpc_mem_handle, which is a pointer to volatile,
             *  and free() doesn't take a pointer to volatile.
             */
            free( (idl_void_p_t)p_thread_support_ptrs->p_mem_h );
            RPC_SS_THREADS_MUTEX_DELETE( &(p_thread_support_ptrs->mutex) );

            /* Free the structure */
            free( p_thread_support_ptrs );
        }

        /* Free the indirection storage */
        free( thread_indirection_ptr );

        /* And destroy the context - this is required for Kernel RPC */
        RPC_SS_THREADS_KEY_SET_CONTEXT( rpc_ss_thread_supp_key, NULL );
    }
}

/******************************************************************************/
/*                                                                            */
/*    rpc_ss_thread_ctx_destructor                                            */
/*    Destroy per thread context at thread termination                        */
/*                                                                            */
/******************************************************************************/
static void rpc_ss_thread_ctx_destructor
(
    rpc_ss_threads_dest_arg_t arg
)
{
    rpc_ss_thread_indirection_t *thread_indirection_ptr =
        (rpc_ss_thread_indirection_t *) arg;

    rpc_ss_destroy_thread_ctx( thread_indirection_ptr );
}

static void rpc_ss_init_allocate(
    void
)
{
    /* Key for thread local storage for tree management */
    RPC_SS_THREADS_KEY_CREATE( &rpc_ss_thread_supp_key,
                    (destructor_t)rpc_ss_thread_ctx_destructor );
}

void rpc_ss_init_allocate_once(
    void
)
{
    RPC_SS_THREADS_INIT;
    RPC_SS_THREADS_ONCE( &allocate_once, rpc_ss_init_allocate );
    rpc_ss_allocate_is_set_up = ndr_true;
}

/******************************************************************************/
/*                                                                            */
/*    Replacement for malloc guaranteed to allocate something like some       */
/*    versions of malloc do.                                                  */
/*                                                                            */
/******************************************************************************/
static void *rpc_ss_client_default_malloc
(
    idl_void_p_t context ATTRIBUTE_UNUSED,
    idl_size_t size
)
{
    void *result = NULL;

    if ( size )
    {
        result = malloc(size);
    }
    else
    {
        result = malloc(1);
    }

    return result;
}

static void rpc_ss_client_default_free
(
    idl_void_p_t context ATTRIBUTE_UNUSED,
    idl_void_p_t ptr
)
{
    free(ptr);
}

/******************************************************************************/
/*                                                                            */
/*    Do we currently have thread context data?                               */
/*    If not, create local storage with malloc() and free() as the            */
/*      allocate and free routines                                            */
/*                                                                            */
/******************************************************************************/
static void rpc_ss_client_get_thread_ctx
(
    rpc_ss_thread_support_ptrs_t **p_p_support_ptrs
)
{
    rpc_ss_thread_support_ptrs_t *p_support_ptrs;
    rpc_ss_thread_indirection_t *thread_indirection_ptr;

#ifndef MEMORY_NOT_WRITTEN_SERIALLY
    if ( ! rpc_ss_allocate_is_set_up )
#endif
        rpc_ss_init_allocate_once();

    p_support_ptrs = (rpc_ss_thread_support_ptrs_t *)rpc_ss_get_thread_handle();
    if (p_support_ptrs == NULL)
    {
        /* We have no context. Make one with the fields we need */
        p_support_ptrs = (rpc_ss_thread_support_ptrs_t *)
                                   malloc(sizeof(rpc_ss_thread_support_ptrs_t));
        if (p_support_ptrs == NULL)
        {
            DCETHREAD_RAISE( rpc_x_no_memory );
        }

        p_support_ptrs->p_mem_h = (rpc_ss_mem_handle *)
                            calloc(1, sizeof(rpc_ss_mem_handle));
        if (p_support_ptrs->p_mem_h == NULL)
        {
            DCETHREAD_RAISE( rpc_x_no_memory );
        }
        p_support_ptrs->p_mem_h->memory = NULL;
        p_support_ptrs->p_mem_h->node_table = NULL;

        RPC_SS_THREADS_MUTEX_CREATE (&(p_support_ptrs->mutex));

        p_support_ptrs->allocator.p_allocate = rpc_ss_client_default_malloc;
        p_support_ptrs->allocator.p_free = rpc_ss_client_default_free;
        p_support_ptrs->allocator.p_context = NULL;

        thread_indirection_ptr = (rpc_ss_thread_indirection_t *)
                        malloc(sizeof(rpc_ss_thread_indirection_t));
        if (thread_indirection_ptr == NULL)
        {
            DCETHREAD_RAISE( rpc_x_no_memory );
        }
        thread_indirection_ptr->indirection = p_support_ptrs;
        thread_indirection_ptr->free_referents = idl_true;
        RPC_SS_THREADS_KEY_SET_CONTEXT( rpc_ss_thread_supp_key,
                                           thread_indirection_ptr );
    }
    *p_p_support_ptrs = p_support_ptrs;
}

/******************************************************************************/
/*                                                                            */
/*    Do we currently have thread context data?                               */
/*    If not, create local storage with malloc() and free() as the            */
/*      allocate and free routines                                            */
/*    Copy pointers to allocate and free routines to local storage            */
/*                                                                            */
/******************************************************************************/
void rpc_ss_client_establish_alloc
(
    rpc_ss_marsh_state_t *p_unmar_params
)
{
    rpc_ss_thread_support_ptrs_t *p_support_ptrs;

    rpc_ss_client_get_thread_ctx( &p_support_ptrs );
    p_unmar_params->allocator = p_support_ptrs->allocator;
}

/******************************************************************************/
/*                                                                            */
/*    rpc_ss_get_thread_handle                                                */
/*                                                                            */
/******************************************************************************/
rpc_ss_thread_handle_t rpc_ss_get_thread_handle
( void )
{
    rpc_ss_thread_indirection_t *thread_indirection_ptr;

    RPC_SS_THREADS_KEY_GET_CONTEXT( rpc_ss_thread_supp_key,
                                       &thread_indirection_ptr );
    if (thread_indirection_ptr == NULL) return(NULL);
    else return (rpc_ss_thread_handle_t)(thread_indirection_ptr->indirection);
}

/******************************************************************************/
/*                                                                            */
/*    rpc_ss_set_thread_handle                                                */
/*                                                                            */
/******************************************************************************/
void rpc_ss_set_thread_handle
(
    rpc_ss_thread_handle_t thread_handle
)
{
    rpc_ss_thread_indirection_t *helper_thread_indirection_ptr;

    /* If a context exists, destroy it */
    RPC_SS_THREADS_KEY_GET_CONTEXT( rpc_ss_thread_supp_key,
                                       &helper_thread_indirection_ptr );
    if ( helper_thread_indirection_ptr != NULL )
    {
        free( helper_thread_indirection_ptr );
    }

    /* Now create the new context */
    helper_thread_indirection_ptr = (rpc_ss_thread_indirection_t *)
                            malloc(sizeof(rpc_ss_thread_indirection_t));
    if (helper_thread_indirection_ptr == NULL)
    {
        DCETHREAD_RAISE( rpc_x_no_memory );
    }
    helper_thread_indirection_ptr->indirection =
                                  (rpc_ss_thread_support_ptrs_t *)thread_handle;
    helper_thread_indirection_ptr->free_referents = idl_false;
    RPC_SS_THREADS_KEY_SET_CONTEXT( rpc_ss_thread_supp_key,
                                           helper_thread_indirection_ptr );
}

/******************************************************************************/
/*                                                                            */
/*    Create thread context with references to named alloc and free rtns      */
/*                                                                            */
/******************************************************************************/
void rpc_ss_set_client_alloc_free
(
    rpc_ss_p_alloc_t p_allocate,
    rpc_ss_p_free_t p_free
)
{
    rpc_ss_allocator_t allocator;

    allocator.p_allocate = p_allocate;
    allocator.p_free = p_free;
    allocator.p_context = NULL;

    rpc_ss_set_client_alloc_free_ex(&allocator);
}

void rpc_ss_set_client_alloc_free_ex
(
    rpc_ss_allocator_t * p_allocator
)
{
    rpc_ss_thread_support_ptrs_t *p_support_ptrs;

    rpc_ss_client_get_thread_ctx( &p_support_ptrs );

    /* Make sure we don't lose a context object. */
    assert(p_support_ptrs->allocator.p_context == NULL);

    p_support_ptrs->allocator = *p_allocator;
}

/******************************************************************************/
/*                                                                            */
/*    Get the existing allocate, free routines and replace them with new ones */
/*                                                                            */
/******************************************************************************/
void rpc_ss_swap_client_alloc_free
(
    rpc_ss_p_alloc_t p_allocate,
    rpc_ss_p_free_t p_free,
    rpc_ss_p_alloc_t *p_p_old_allocate,
    rpc_ss_p_free_t * p_p_old_free
)
{
    rpc_ss_allocator_t allocator;

    allocator.p_allocate = p_allocate;
    allocator.p_free = p_free;
    allocator.p_context = NULL;

    rpc_ss_swap_client_alloc_free_ex(&allocator, &allocator);

    /* Make sure we didn't lose a context object. */
    assert(allocator.p_context == NULL);

    *p_p_old_allocate = allocator.p_allocate;
    *p_p_old_free = allocator.p_free;
}

void rpc_ss_swap_client_alloc_free_ex
(
    rpc_ss_allocator_t * p_allocator,
    rpc_ss_allocator_t * p_old_allocator
)
{
    rpc_ss_allocator_t allocator;
    rpc_ss_thread_support_ptrs_t *p_support_ptrs;

    rpc_ss_client_get_thread_ctx( &p_support_ptrs );

    /* Make sure it is safe for the old and new pointers to point
     * to the same allocator.
     */
    allocator = *p_allocator;
    *p_old_allocator = p_support_ptrs->allocator;
    p_support_ptrs->allocator = allocator;
}

/****************************************************************************/
/* rpc_ss_client_free                                                       */
/*                                                                          */
/* Free the specified memory using the free routine from the current memory */
/* management environment.  This routine provides a simple interface to     */
/* free memory returned from an RPC call.                                   */
/*                                                                          */
/****************************************************************************/
void rpc_ss_client_free
(
    idl_void_p_t p_mem
)
{
    rpc_ss_thread_support_ptrs_t *p_support_ptrs;

    /* Get free routine address */
    rpc_ss_client_get_thread_ctx( &p_support_ptrs );
    /* Invoke free with the specified memory */
    rpc_allocator_free(&p_support_ptrs->allocator, p_mem);
}

/******************************************************************************/
/*                                                                            */
/*    rpc_ss_enable_allocate                                                  */
/*    Create environment for rpc_ss_allocate to be used                       */
/*                                                                            */
/******************************************************************************/
void rpc_ss_enable_allocate
( void )
{
    rpc_ss_mem_handle *p_mem_handle;
    rpc_ss_thread_support_ptrs_t *p_thread_support_ptrs;

    /* Make sure there is a thread context key */
#ifndef MEMORY_NOT_WRITTEN_SERIALLY
    if ( ! rpc_ss_allocate_is_set_up )
#endif
        rpc_ss_init_allocate_once();

    /* Set up the parts of the required data structure */
    p_mem_handle = (rpc_ss_mem_handle *)malloc(sizeof(rpc_ss_mem_handle));
    if (p_mem_handle == NULL)
    {
        DCETHREAD_RAISE( rpc_x_no_memory );
    }
    p_mem_handle->memory = NULL;
    p_mem_handle->node_table = NULL;
    p_thread_support_ptrs = (rpc_ss_thread_support_ptrs_t *)
                                   malloc(sizeof(rpc_ss_thread_support_ptrs_t));
    if (p_thread_support_ptrs == NULL)
    {
        DCETHREAD_RAISE( rpc_x_no_memory );
    }

    /* Complete the data structure and associate it with the key */
    /* This will make rpc_ss_allocate, rpc_ss_free the allocate/free pair */
    rpc_ss_build_indirection_struct( p_thread_support_ptrs, p_mem_handle,
                                     idl_true );
}

/******************************************************************************/
/*                                                                            */
/*    rpc_ss_disable_allocate                                                 */
/*    Destroy environment created by rpc_ss_enable_allocate                   */
/*                                                                            */
/******************************************************************************/
void rpc_ss_disable_allocate
( void )
{
    rpc_ss_thread_indirection_t *helper_thread_indirection_ptr;

    /* Get the thread support pointers structure */
    RPC_SS_THREADS_KEY_GET_CONTEXT( rpc_ss_thread_supp_key,
                                       &helper_thread_indirection_ptr );

    rpc_ss_destroy_thread_ctx( helper_thread_indirection_ptr );
}

#ifdef MIA
/******************************************************************************/
/*                                                                            */
/*    MTS version of                                                          */
/*                                                                            */
/*    Do we currently have thread context data?                               */
/*    If not, create local storage with malloc() and free() as the            */
/*      allocate and free routines                                            */
/*    Copy pointers to allocate and free routines to local storage            */
/*                                                                            */
/******************************************************************************/
void rpc_ss_mts_client_estab_alloc
(
    volatile IDL_ms_t * IDL_msp
)
{
    rpc_ss_thread_support_ptrs_t *p_support_ptrs;

#ifdef PERFMON
    RPC_SS_CLIENT_ESTABLISH_ALLOC_N;
#endif

    rpc_ss_client_get_thread_ctx( &p_support_ptrs );
    IDL_msp->IDL_allocator = p_support_ptrs->allocator;

#ifdef PERFMON
    RPC_SS_CLIENT_ESTABLISH_ALLOC_X;
#endif

}
#endif

