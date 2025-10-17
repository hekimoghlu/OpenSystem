/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 29, 2021.
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
**      comstmts.c
**
**  FACILITY:
**
**      Interface Definition Language (IDL) Compiler
**
**  ABSTRACT:
**
**      Deliver comm_status or raise exception in client stub
**      Deliver fault_status or raise exception in client stub
**      For MTS compiler
**
*/

#include <nidl.h>
#include <ast.h>
#include <bedeck.h>
#include <commstat.h>
#include <cspell.h>
#include <nametbl.h>

/*******************************************************************************/
/*                                                                             */
/*  Determine how comm_status is to be returned                                */
/*                                                                             */
/*******************************************************************************/
void BE_get_comm_stat_info
(
    AST_operation_n_t *p_operation,
    BE_stat_info_t *p_comm_stat_info
)
{
    AST_parameter_n_t *p_parameter;

    if ( AST_COMM_STATUS_SET(p_operation->result) )
    {
        p_comm_stat_info->type = BE_stat_result_k;
        return;
    }
    for ( p_parameter = p_operation->parameters;
          p_parameter != NULL;
          p_parameter = p_parameter->next )
    {
        if ( AST_COMM_STATUS_SET(p_parameter) )
        {
            p_comm_stat_info->type = BE_stat_param_k;
            p_comm_stat_info->name = p_parameter->name;
            return;
        }
        else if ( AST_ADD_COMM_STATUS_SET(p_parameter) )
        {
            p_comm_stat_info->type = BE_stat_addl_k;
            p_comm_stat_info->name = p_parameter->name;
            return;
        }
    }
    p_comm_stat_info->type = BE_stat_except_k;
}

/*******************************************************************************/
/*                                                                             */
/*  Determine how fault_status is to be returned                               */
/*                                                                             */
/*******************************************************************************/
void BE_get_fault_stat_info
(
    AST_operation_n_t *p_operation,
    BE_stat_info_t *p_fault_stat_info
)
{
    AST_parameter_n_t *p_parameter;

    if ( AST_FAULT_STATUS_SET(p_operation->result) )
    {
        p_fault_stat_info->type = BE_stat_result_k;
        return;
    }
    for ( p_parameter = p_operation->parameters;
          p_parameter != NULL;
          p_parameter = p_parameter->next )
    {
        if ( AST_FAULT_STATUS_SET(p_parameter) )
        {
            p_fault_stat_info->type = BE_stat_param_k;
            p_fault_stat_info->name = p_parameter->name;
            return;
        }
        else if ( AST_ADD_FAULT_STATUS_SET(p_parameter) )
        {
            p_fault_stat_info->type = BE_stat_addl_k;
            p_fault_stat_info->name = p_parameter->name;
            return;
        }
    }
    p_fault_stat_info->type = BE_stat_except_k;
}

/*******************************************************************************/
/*                                                                             */
/*    Spell code that returns status to client                                 */
/*                                                                             */
/*******************************************************************************/
void CSPELL_return_status
(
    FILE *fid,
    BE_stat_info_t *p_comm_stat_info,
    BE_stat_info_t *p_fault_stat_info,
    const char *status_var_name,
    const char *result_param_name,
    int num_user_exceptions,
    const char *IDL_msp_name     /* Lexical form of pointer to IDL_ms_t state block */
)
{
#define MAX_STATUS_STRING 72+MAX_ID+MAX_ID
    char const *str_p_comm_status; /* String used as parameter describing how
                                    comm status is to be returned */
    char const *str_p_fault_status;/* String used as parameter describing how
                                    fault status is to be returned */
    char const *name_work;
    char comm_status_work[MAX_STATUS_STRING];
    char fault_status_work[MAX_STATUS_STRING];

    /* Handle [comm_status] parameters */
    switch( p_comm_stat_info->type )
    {
        case BE_stat_addl_k:
            /*
             *  If an added comm_status parameter, always pass it to
             *  rpc_ss_report_error so that it will be set to either
             *  error_status_ok, or the correct status value.
             */
            NAMETABLE_id_to_string( p_comm_stat_info->name,
                                                       &str_p_comm_status );
            break;
        case BE_stat_result_k:
            /*
             * For a function result, pass the address of the surrogate, if
             * we had a comm_status-related error.  Otherwise pass NULL so
             * that rpc_ss_report_error will not overwrite the user's value
             */
            sprintf( comm_status_work,"(%s!=error_status_ok) ? &%s : NULL",
                        status_var_name, result_param_name );
            str_p_comm_status = comm_status_work;
            break;
        case BE_stat_param_k:
            /*
             *  For a user parameter, pass the parameter, if we had a
             *  comm_status-related error.  Otherwise pass NULL so that
             *  rpc_ss_report_error will not overwrite the user's value
             */
            NAMETABLE_id_to_string( p_comm_stat_info->name, &name_work );
            sprintf( comm_status_work,"(%s!=error_status_ok) ? %s : NULL",
                        status_var_name, name_work );
            str_p_comm_status = comm_status_work;
            break;
        default:
            str_p_comm_status = "NULL";
            break;
    }
    ASSERTION(strlen(str_p_comm_status) < MAX_STATUS_STRING);

    /* Handle [fault_status] parameters */
    switch( p_fault_stat_info->type )
    {
        case BE_stat_addl_k:
            /*
             *  If an added fault_status parameter, always pass it to
             *  rpc_ss_report_error so that it will be set to either
             *  error_status_ok, or the correct status value.
             */
            NAMETABLE_id_to_string( p_fault_stat_info->name,
                                                       &str_p_fault_status );
            break;
        case BE_stat_result_k:
            /*
             *  For a function result, pass the address of the surrogate, if we
             *  had an error.  Otherwise pass NULL so that rpc_ss_report_error
             *  will not overwrite the user's value.  We need not distiguish
             *  between comm/fault errors because either will mean that the
             *  user's values has not been unmarshalled anyway and thus we
             *  should set it to error_status_ok.
             */
            sprintf( fault_status_work,"(%s!=error_status_ok) ? &%s : NULL",
                        status_var_name, result_param_name );
            str_p_fault_status = fault_status_work;
            break;
        case BE_stat_param_k:
            /*
             *  For a function result, pass the address of the surrogate, if we
             *  had an error.  Otherwise pass NULL so that rpc_ss_report_error
             *  will not overwrite the user's value.  We need not distiguish
             *  between comm/fault errors because either will mean that the
             *  user's values has not been unmarshalled anyway and thus we
             *  should set it to error_status_ok.
             */
            NAMETABLE_id_to_string( p_fault_stat_info->name, &name_work );
            sprintf( fault_status_work,"(%s!=error_status_ok) ? %s : NULL",
                        status_var_name, name_work );
            str_p_fault_status = fault_status_work;
            break;
        default:
            str_p_fault_status = "NULL";
            break;
    }
    ASSERTION(strlen(str_p_fault_status) < MAX_STATUS_STRING);

    fprintf( fid,
                "rpc_ss_report_error_2(IDL_fault_code,IDL_user_fault_id,%s,\n",
                      status_var_name );
    fprintf( fid,
" (RPC_SS_THREADS_CANCEL_STATE_T *)&IDL_async_cancel_state, %s, %s, %s, %s);\n",
             str_p_comm_status, str_p_fault_status,
             num_user_exceptions ? "IDL_exception_addresses" : "NULL",
             IDL_msp_name
             );

}
