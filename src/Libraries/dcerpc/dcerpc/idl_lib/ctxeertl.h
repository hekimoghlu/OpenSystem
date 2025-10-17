/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 28, 2023.
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
**      ctxeertl.h
**
**  FACILITY:
**
**      IDL Stub Runtime Support
**
**  ABSTRACT:
**
**      Definitions for callee stub's tables of contexts and clients
**
**  VERSION: DCE 1.0
**
**
*/

/*  The structure of the callee client lookup table
**  is overflow hash with double chaining.
*/
typedef struct callee_client_entry_t {
    rpc_client_handle_t client;
    long int count;    /* Number of contexts for this client */
    struct callee_context_entry_t *first_context; /* First in chain of
                                                     contexts for this client */
    struct callee_context_entry_t *last_context;  /* Last in chain of
                                                     contexts for this client */
    struct callee_client_entry_t *prev_h_client;    /* Previous in chain of
                                                clients with same hash value */
    struct callee_client_entry_t *next_h_client;    /* Next in chain of
                                                clients with same hash value */
    long int ref_count;  /* The number of threads currently using contexts
                                of this client */
    RPC_SS_THREADS_CONDITION_T cond_var;  /* Used to signal when ref_count
                                                has been decremented */
    idl_boolean rundown_pending;    /* TRUE if a rundown request for this
                                        client has been received */
} callee_client_entry_t;

/*  The structure of the callee context lookup table
**  is overflow hash with chaining.
*/
typedef struct callee_context_entry_t {
    idl_uuid_t uuid;
    rpc_ss_context_t user_context;
    ctx_rundown_fn_p_t rundown; /* Pointer to rundown routine for context */
    callee_client_entry_t *p_client_entry;  /* Client this context belongs to */
    struct callee_context_entry_t *prev_in_client;  /* Previous in chain of
                                                        contexts for client */
    struct callee_context_entry_t *next_in_client;  /* Next in chain of
                                                        contexts for client */
    struct callee_context_entry_t *next_context;    /* Next in chain of
                                            contexts with the same hash value */
} callee_context_entry_t;

/**************** Function prototypes *******************************/

void rpc_ss_rundown_client
(
    rpc_client_handle_t failed_client
);

void rpc_ss_add_to_callee_client
(
    rpc_client_handle_t ctx_client,     /* Client for whom there is another context */
    callee_context_entry_t *p_context,  /* Pointer to the context */
    ndr_boolean *p_is_new_client,       /* Pointer to TRUE if new client */
    error_status_t *result         /* Function result */
);

void rpc_ss_take_from_callee_client
(
    callee_context_entry_t *p_context,  /* Pointer to the context */
    rpc_client_handle_t *p_close_client,
                                  /* Ptr to NULL or client to stop monitoring */
    error_status_t *result         /* Function result */
);

void rpc_ss_lkddest_callee_context
(
    idl_uuid_t *p_uuid,    /* Pointer to UUID of context to be destroyed */
    rpc_client_handle_t *p_close_client,
                         /* Ptr to NULL or client to stop monitoring */
    error_status_t *result         /* Function result */
);    /* Returns SUCCESS unless the UUID is not in the lookup table */

void rpc_ss_init_callee_client_table(
    void
);

void rpc_ss_create_callee_context
(
    rpc_ss_context_t callee_context, /* user's local form of the context */
    idl_uuid_t    *p_uuid,               /* pointer to the equivalent UUID */
    handle_t  h,                     /* Binding handle */
    ctx_rundown_fn_p_t ctx_rundown,  /* pointer to context rundown routine */
    error_status_t *result      /* Function result */
);

/* Returns status_ok unless the UUID is not in the lookup table */
void rpc_ss_update_callee_context
(
    rpc_ss_context_t callee_context, /* user's local form of the context */
    idl_uuid_t    *p_uuid,               /* pointer to the equivalent UUID */
    error_status_t *result      /* Function result */
);

/* Returns status_ok unless the UUID is not in the lookup table */
void rpc_ss_destroy_callee_context
(
    idl_uuid_t *p_uuid,          /* pointer to UUID of context to be destroyed */
    handle_t  h,                    /* Binding handle */
    error_status_t *result     /* Function result */
);
